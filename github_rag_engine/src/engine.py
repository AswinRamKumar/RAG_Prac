
import os
import hashlib
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    Settings,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.storage.kvstore import SimpleKVStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate, SimpleDirectoryReader, Document

from . import config
from .utils import filesystem
from .ingestion import chunking

class GitHubRAG:
    def __init__(self):
        Settings.embed_model = OpenAIEmbedding(model=config.EMBED_MODEL)
        Settings.llm = OpenAI(model=config.OPENAI_MODEL, temperature=0.1)
        
        self.db_path = str(config.CHROMA_PATH)
        # We will load these lazily when a repo is selected
        self.active_repo = None
        self.chroma_client = None
        self.collection = None
        self.index = None
        self.query_engine = None
        
        # Ensure base directories exist
        os.makedirs(config.REPO_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(config.INGESTION_CACHE), exist_ok=True)
        
        print(f"🔌 Connecting to ChromaDB at {self.db_path}...")
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)

    def list_repos(self):
        """Returns a list of available repo names from disk."""
        try:
            return [d for d in os.listdir(config.REPO_DIR) 
                   if os.path.isdir(os.path.join(config.REPO_DIR, d)) and not d.startswith(".")]
        except:
            return []

    def initialize_repo(self, repo_name: str):
        """Switches the active collection to the specified repo."""
        if self.active_repo == repo_name and self.index is not None:
             return # Already loaded
             
        print(f"🔄 Switching to repo: {repo_name}")
        self.active_repo = repo_name
        
        # 1. Get/Create Collection specific to this repo
        # Format: "repo_Diff-Retinex-Plus"
        safe_name = f"repo_{repo_name}".replace(" ", "_").replace(".", "_")
        self.collection = self.chroma_client.get_or_create_collection(safe_name)
        
        # --- LEGACY MIGRATION CHECK ---
        if self.collection.count() == 0:
            try:
                legacy_coll = self.chroma_client.get_collection("github_rag_v1")
                if legacy_coll.count() > 0:
                    print(f"📦 Found legacy 'github_rag_v1' with {legacy_coll.count()} docs. Migrating usage...")
                    self.collection = legacy_coll
            except ValueError:
                pass
        
        # 2. Load Index (if exists)
        if self.collection.count() > 0:
            print(f"✅ Found {self.collection.count()} chunks. Loading index...")
            vector_store = ChromaVectorStore(chroma_collection=self.collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context
            )
            # Reset query engine so it rebuilds with new index
            self.query_engine = None
        else:
            print("⚠️ Collection empty. awaiting ingestion.")
            self.index = None

    def ingest_repo(self, repo_url: str):
        """Ingests a new repo into its own collection."""
        # 1. Clone
        repo_path = filesystem.clone_repo(repo_url)
        repo_name = repo_path.name
        
        # 2. Switch to this repo (creates the collection)
        self.initialize_repo(repo_name)
        
        # 3. Check if already indexed
        if self.collection.count() > 0:
             print("ℹ️ Repo already indexed. Skipping processing.")
             return
             
        # 4. Load Files
        source_files = filesystem.get_source_files(repo_path)
        if not source_files:
            return

        # 5. Repo Map
        repo_map_str = filesystem.generate_repo_map(repo_path)
        map_doc = Document(
            text=repo_map_str,
            metadata={"file_path": "REPO_STRUCTURE.txt", "type": "structure"}
        )
        
        # 6. Read & Chunk
        reader = SimpleDirectoryReader(
            input_files=[str(f) for f in source_files],
            file_metadata=chunking.file_metadata_extractor
        )
        docs = reader.load_data()
        all_docs = [map_doc] + docs
        nodes = chunking.get_nodes_adaptive(all_docs)
        
        # 7. Embed (with Cache)
        # We share one ingestion cache across all repos to save money on identical files
        if os.path.exists(config.INGESTION_CACHE):
            kv_store = SimpleKVStore.from_persist_path(str(config.INGESTION_CACHE))
        else:
            kv_store = SimpleKVStore()
            
        pipeline = IngestionPipeline(
            transformations=[Settings.embed_model],
            cache=IngestionCache(cache=kv_store)
        )
        
        nodes_with_embeddings = pipeline.run(nodes=nodes)
        kv_store.persist(str(config.INGESTION_CACHE))
        
        # 8. Index
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex(
            nodes_with_embeddings,
            storage_context=storage_context
        )
        print("✅ Ingestion & Indexing Complete!")

    def query(self, question: str):
        if not self.index:
            raise ValueError("No active index! Select or ingest a repo first.")
            
        if not self.query_engine:
            self._build_query_engine()
            
        return self.query_engine.query(question)
        
    def _build_query_engine(self):
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=10)
        post_processor = SimilarityPostprocessor(similarity_cutoff=0.1)
        
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[post_processor]
        )

    def get_indexed_count(self):
        """Returns the number of chunks currently in the vector store."""
        if self.collection:
            return self.collection.count()
        return 0
        
    def get_active_repo(self):
        """Returns the name of the active repo."""
        if self.active_repo:
             return self.active_repo
        return "None"

if __name__ == "__main__":
    engine = GitHubRAG()
    # engine.ingest_repo("...")

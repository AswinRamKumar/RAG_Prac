
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
        self.active_repo = None
        self.chroma_client = None
        self.collection = None
        self.index = None
        self.query_engine = None
        
        os.makedirs(config.REPO_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(config.INGESTION_CACHE), exist_ok=True)
        
        print(f"🔌 Connecting to ChromaDB at {self.db_path}...")
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)

    def list_repos(self):
        try:
            return [d for d in os.listdir(config.REPO_DIR) 
                   if os.path.isdir(os.path.join(config.REPO_DIR, d)) and not d.startswith(".")]
        except:
            return []

    def initialize_repo(self, repo_name: str):
        self.index = None
        self.query_engine = None
        self.collection = None
        self.active_repo = repo_name

        print(f"🔄 Switching to repo: {repo_name}")

        safe_name = f"repo_{repo_name}".replace(" ", "_").replace(".", "_")

        try:
            self.collection = self.chroma_client.get_collection(safe_name)
            print("✅ Existing collection found.")
        except Exception:
            print("⚠️ Collection not found. Creating new one.")
            self.collection = self.chroma_client.create_collection(safe_name)

        if self.collection.count() > 0:
            print(f"✅ Found {self.collection.count()} chunks. Loading index...")
            vector_store = ChromaVectorStore(chroma_collection=self.collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            self.index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context
            )
        else:
            print("⚠️ Collection empty. Awaiting ingestion.")
            self.index = None


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
        if self.collection:
            return self.collection.count()
        return 0
        
    def get_active_repo(self):
        if self.active_repo:
             return self.active_repo
        return "None"

if __name__ == "__main__":
    engine = GitHubRAG()

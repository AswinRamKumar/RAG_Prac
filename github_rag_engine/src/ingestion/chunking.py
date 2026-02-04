
import os
import tiktoken
from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import CodeSplitter, MarkdownNodeParser, SentenceSplitter
from src import config

def file_metadata_extractor(file_path: str) -> dict:
    return {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "extension": os.path.splitext(file_path)[1],
    }

def get_nodes_adaptive(documents: List[Document]) -> List[Document]:
    all_nodes = []
    
    tokenizer = tiktoken.encoding_for_model(config.OPENAI_MODEL).encode
    
    py_splitter = CodeSplitter(
        language="python", 
        chunk_lines=150, 
        chunk_lines_overlap=30, 
        max_chars=3000
    )
    
    md_splitter = MarkdownNodeParser(
        chunk_size=800,
        chunk_overlap=100
    )
    
    text_splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
        tokenizer=tokenizer
    )
    
    print(f"🧩 Chunking {len(documents)} documents...")
    for doc in documents:
        ext = doc.metadata.get("extension", "").lower()
        fn = doc.metadata.get("file_name", "")
        
        if ext == ".py":
            nodes = py_splitter.get_nodes_from_documents([doc])
        elif ext == ".md":
            nodes = md_splitter.get_nodes_from_documents([doc])
        else:
            nodes = text_splitter.get_nodes_from_documents([doc])
            
        for i, n in enumerate(nodes):
            n.metadata["chunk_id"] = i
            n.metadata["file_name"] = fn
            n.metadata["file_ext"] = ext
            
        all_nodes.extend(nodes)
        
    return all_nodes

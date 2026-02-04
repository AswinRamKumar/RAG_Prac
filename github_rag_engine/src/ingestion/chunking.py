
import os
import tiktoken
from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import CodeSplitter, MarkdownNodeParser, SentenceSplitter
from src import config

# Map extensions to tree-sitter languages
# Ensure tree-sitter-languages is installed for these to work
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".h": "cpp",
    ".c": "c",
    ".cs": "c_sharp", 
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".scala": "scala",
    ".swift": "swift", 
    ".kt": "kotlin",
}

def file_metadata_extractor(file_path: str) -> dict:
    return {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "extension": os.path.splitext(file_path)[1],
    }

def get_nodes_adaptive(documents: List[Document]) -> List[Document]:
    all_nodes = []
    
    tokenizer = tiktoken.encoding_for_model(config.OPENAI_MODEL).encode
    
    # Generic text splitter for fallback
    sentence_splitter = SentenceSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        tokenizer=tokenizer
    )
    
    # Markdown specific
    md_splitter = MarkdownNodeParser(
        chunk_size=800,
        chunk_overlap=100
    )
    
    for doc in documents:
        ext = doc.metadata.get("extension", "").lower()
        
        # 1. Markdown
        if ext == ".md":
            all_nodes.extend(md_splitter.get_nodes_from_documents([doc]))
            
        # 2. Supported Code Language
        elif ext in EXTENSION_TO_LANGUAGE:
            lang = EXTENSION_TO_LANGUAGE[ext]
            try:
                code_splitter = CodeSplitter(
                    language=lang,
                    chunk_lines=150,
                    chunk_lines_overlap=30,
                    max_chars=3000
                )
                all_nodes.extend(code_splitter.get_nodes_from_documents([doc]))
            except Exception as e:
                # If tree-sitter grammar missing, fallback
                print(f"⚠️ Code split warning for {doc.metadata['file_name']} ({lang}): {e}. Using fallback.")
                all_nodes.extend(sentence_splitter.get_nodes_from_documents([doc]))
                
        # 3. Everything else (Text, unknown code)
        else:
            all_nodes.extend(sentence_splitter.get_nodes_from_documents([doc]))
            
    return all_nodes

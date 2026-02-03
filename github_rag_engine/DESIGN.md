# Production RAG Design Doc

## 1. Core Philosophy

- **Separation of Concerns**: Ingestion and Querying are decoupled.
- **Local First**: We clone locally to allow deep static analysis if needed later.
- **Explainability**: We want to know _which_ files were retrieved and why.

## 2. Ingestion Pipeline

1. **Cloner**: Takes a GitHub URL -> Clones to `./data/repos/<owner>_<repo>`.
2. **FileWalker**: Recursively walks the directory.
   - **Constraint**: Must respect `.gitignore`.
   - **Constraint**: Must exclude binary/lock files.
3. **Chunker**: Code-specific chunking (not just character splitting).
   - _Future_: Tree-sitter parsing for class/function boundaries.

## 3. Storage

- **Vector DB**: ChromaDB (Run locally for now, easy to swap).
- **Metadata**: Every chunk needs: `filepath`, `line_start`, `line_end`, `commit_hash`.


import streamlit as st
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.engine import GitHubRAG
from src import config
from src.utils import filesystem

st.set_page_config(page_title="GitHub RAG", page_icon="🤖", layout="wide")

st.title("🤖 GitHub Code Assistant")

# Initialize Engine
if "rag" not in st.session_state:
    st.session_state.rag = GitHubRAG()

# Sidebar: Repo Management
with st.sidebar:
    st.header("📂 Repository Manager")
    
    # 1. Select Active Repo
    available_repos = st.session_state.rag.list_repos()
    
    if available_repos:
        selected_repo = st.selectbox("Select Active Repo", available_repos)
        
        if selected_repo:
             # Initialize the backend for this repo
             st.session_state.rag.initialize_repo(selected_repo)
             
             # Check Status
             idx_count = st.session_state.rag.get_indexed_count()
             
             if idx_count > 0:
                 st.success(f"✅ Ready: {idx_count} chunks")
             else:
                 st.warning("⚠️ Index Empty/Missing")
                 
                 # AUTO-REPAIR Logic
                 repo_path = config.REPO_DIR / selected_repo
                 detected_url = filesystem.get_repo_url(repo_path)
                 
                 if detected_url:
                     st.info(f"Detected remote: {detected_url}")
                     if st.button("⚡ Re-Build Index"):
                         with st.spinner("Re-indexing... (this is fast)"):
                             st.session_state.rag.ingest_repo(detected_url)
                             st.rerun()
                 else:
                     st.error("Could not detect URL. Delete folder or re-add below.")

    else:
        st.info("No repositories found.")

    st.divider()
    
    # 2. Add New Repo
    st.subheader("➕ Add New Repository")
    new_repo_url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo")
    
    if st.button("Clone & Ingest"):
        if new_repo_url:
            with st.status("Processing...", expanded=True) as status:
                st.write("Cloning...")
                try:
                    st.session_state.rag.ingest_repo(new_repo_url)
                    status.update(label="✅ Complete!", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(f"Failed: {e}")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat only if an index is valid
if st.session_state.rag.index:
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input(f"Ask about {st.session_state.rag.active_repo}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag.query(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": str(response)})
                except Exception as e:
                    st.error(f"Error: {e}")
else:
    if available_repos:
        st.info("👈 Please repair the index in the sidebar to start chatting.")
    else:
        st.info("👈 Please add a repository to start.")

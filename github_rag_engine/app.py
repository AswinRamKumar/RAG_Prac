
import streamlit as st
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.engine import GitHubRAG

st.set_page_config(page_title="GitHub RAG", page_icon="🤖")

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
             st.session_state.rag.initialize_repo(selected_repo)
             # Show Stats
             if st.session_state.rag.collection:
                 count = st.session_state.rag.collection.count()
                 if count > 0:
                     st.success(f"✅ Ready: {count} chunks")
                 else:
                     st.warning("⚠️ Index empty. Please Re-Index.")
                     if st.button("Build Index for this Repo"):
                         with st.spinner("Indexing..."):
                             # We can trigger ingest logic with the folder name?
                             # Actually ingest_repo usually takes a URL.
                             # But if folder exists, we can reconstruct a dummy URL or just separate logic.
                             # Let's just ask user to paste URL to be safe/simple.
                             st.info("Please paste the URL below to index.")
    else:
        st.info("No repositories found. Add one below.")

    st.divider()
    
    # 2. Add New Repo
    st.subheader("➕ Add New Repository")
    new_repo_url = st.text_input("GitHub URL")
    
    if st.button("Clone & Ingest"):
        if new_repo_url:
            with st.status("Processing...", expanded=True) as status:
                st.write("Cloning...")
                try:
                    st.session_state.rag.ingest_repo(new_repo_url)
                    status.update(label="✅ Compete! Repo added.", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(f"Failed: {e}")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat only if an index is loaded
if st.session_state.rag.index:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the active repo..."):
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
    st.info("👈 Please select or ingest a repository to start chatting.")

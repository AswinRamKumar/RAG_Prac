import streamlit as st
import sys
import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from src.engine import GitHubRAG
from src import config
from src.utils import filesystem

st.set_page_config(
    page_title="GitHub RAG",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 GitHub Code Assistant")

if "rag" not in st.session_state:
    st.session_state.rag = GitHubRAG()

rag = st.session_state.rag

with st.sidebar:
    st.header("📂 Repository Manager")

    available_repos = rag.list_repos()

    if available_repos:
        selected_repo = st.selectbox(
            "Select Active Repo",
            available_repos,
            key="active_repo_select",
        )

        if selected_repo and rag.get_active_repo() != selected_repo:
            rag.initialize_repo(selected_repo)

        idx_count = rag.get_indexed_count()

        if idx_count > 0:
            st.success(f"✅ Ready: {idx_count} chunks")
        else:
            st.warning("⚠️ Index Empty / Missing")

            repo_path = config.REPO_DIR / selected_repo
            detected_url = filesystem.get_repo_url(repo_path)

            if detected_url:
                st.info(f"Detected remote: {detected_url}")

                if st.button("⚡ Re-build Index", key="repair_index"):
                    with st.spinner("Re-indexing repository..."):
                        rag.ingest_repo(detected_url)
                    st.rerun()
            else:
                st.error(
                    "Remote URL not detected. "
                    "Delete the repo folder and re-add."
                )
    else:
        st.info("No repositories found.")

    st.divider()

    st.subheader("➕ Add New Repository")

    new_repo_url = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/owner/repo",
    )

    if st.button("Clone & Ingest", key="clone_ingest"):
        if not new_repo_url.strip():
            st.error("Please enter a GitHub repository URL.")
        else:
            with st.status("Processing repository...", expanded=True) as status:
                try:
                    st.write("📥 Cloning & ingesting...")
                    rag.ingest_repo(new_repo_url)
                    status.update(
                        label="✅ Ingestion complete!",
                        state="complete",
                        expanded=False,
                    )
                    st.rerun()
                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(str(e))

if "messages" not in st.session_state:
    st.session_state.messages = []

if rag.index:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input(
        f"Ask about {rag.get_active_repo()}..."
    ):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = rag.query(prompt)
                    st.markdown(response)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": str(response),
                        }
                    )
                except Exception as e:
                    st.error(f"Query failed: {e}")
else:
    if available_repos:
        st.info("👈 Select or repair a repository to start chatting.")
    else:
        st.info("👈 Add a repository to begin.")

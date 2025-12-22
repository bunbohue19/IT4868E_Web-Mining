import sys
from pathlib import Path

import chromadb
import streamlit as st
from langchain_chroma import Chroma

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedding.embedding_model import ONNXEmbeddings


def build_vector_store() -> Chroma:
    embeddings = ONNXEmbeddings(model_path=f"{PROJECT_ROOT}/model/gte-multilingual-base-JA_onnx")
    persist_directory = f"{PROJECT_ROOT}/vector_database/words"
    persistent_client = chromadb.PersistentClient(path=persist_directory)
    return Chroma(
        client=persistent_client,
        collection_name="word_vector",
        embedding_function=embeddings,
    )


def lookup(vector_store: Chroma, query: str, k: int = 1):
    return vector_store.similarity_search(query, k=k)


@st.cache_resource(show_spinner=False)
def get_vector_store() -> Chroma:
    # Cached so Streamlit does not re-load the model/vector DB on each interaction.
    return build_vector_store()


def render_results(results):
    for idx, doc in enumerate(results, start=1):
        formatted = (doc.metadata or {}).get("formatted")
        st.markdown(f"**Result {idx}: {doc.page_content}**")
        if formatted:
            st.code(formatted, language="text")


def main():
    st.title("Dictionary Lookup")
    st.caption("Search the Chroma store built from dataset/words")

    query = st.text_input("Enter a query", "")
    k = st.slider("Number of results", min_value=1, max_value=10, value=3, step=1)

    if not query:
        st.info("Enter a query to search the vector database.")
        return

    try:
        vector_store = get_vector_store()
    except Exception as exc:
        st.error("Could not load the vector database. Ensure it has been built via indexing.")
        st.exception(exc)
        return

    with st.spinner("Searching..."):
        results = lookup(vector_store, query, k=k)

    if not results:
        st.warning("No results found.")
        return

    render_results(results)


if __name__ == "__main__":
    main()
import sys
import chromadb
from pathlib import Path
from langchain_chroma import Chroma

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedding.embedding_model import ONNXEmbeddings

def lookup(vector_store: Chroma, query: str, k=1):
    return vector_store.similarity_search(query, k=k)

def main():
    # 1. Initialize Text Embedding
    embeddings = ONNXEmbeddings(model_path=f"{PROJECT_ROOT}/model/gte-multilingual-base-JA_onnx")

    # 2. Indexing with a vector database by embedding then inserting document chunks
    # Locate vector database
    persist_directory = f"{PROJECT_ROOT}/vector_database/words"
    persistent_client = chromadb.PersistentClient(path=persist_directory)
    vector_store = Chroma(
        client=persistent_client,
        collection_name="word_vector",
        embedding_function=embeddings
    )
    
    # Example query
    query = "Bố"
    
    print(lookup(vector_store, query, k=1))

if __name__ == "__main__":
    main()
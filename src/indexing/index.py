import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from indexing.record.word import Word
from embedding.embedding_model import ONNXEmbeddings


print(PROJECT_ROOT)

def main():
    record_type = "word"
    word = Word(file_path=f"{PROJECT_ROOT}/dataset/words/[JA-JA]_大辞泉_第二版.jsonl")
    word_loader = word.init_loader()
    data = word_loader.load()
    
    # 1. Data Preparation for Retrieval
    text_splitter = CharacterTextSplitter(chunk_size=4000)
    chunks = text_splitter.split_documents(data)

    # Initialize Text Embedding
    embeddings = ONNXEmbeddings(model_path=f"{PROJECT_ROOT}/model/gte-multilingual-base-JA_onnx")
    
    # 2. Indexing with a vector database by embedding then inserting document chunks
    # Locate vector database
    persist_directory = f"{PROJECT_ROOT}/vector_database/words"
    
    # Create vector database
    batch_size = 41666
    for i in tqdm(range(0, len(chunks), batch_size), desc=f"Processing {record_type}"):
        batch = chunks[i : i + batch_size]
        _ = Chroma.from_documents(
            documents=batch, 
            embedding=embeddings, 
            collection_name=f"{record_type}_vector",
            persist_directory=persist_directory
        )

if __name__ == "__main__":
    main()
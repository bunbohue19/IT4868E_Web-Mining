import chromadb
from pathlib import Path
from tqdm import tqdm
from record.word import Word
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from src.embedding.embedding_model import ONNXEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[2]

print(PROJECT_ROOT)

def main():
    word = Word()
    word_loader = word.init_loader()
    data = word_loader.load()
    
    # 1. Data Preparation for Retrieval
    text_splitter = CharacterTextSplitter(chunk_size=200)
    chunks = text_splitter.split_documents(data)
    
    # Retrieve embedding function from code env resources
    model_kwargs = {'device': 'cuda:0'}
    encode_kwargs = {"normalize_embeddings": True, "batch_size": 32}

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
            collection_name="word_vector",
            persist_directory=persist_directory
        )

if __name__ == "__main__":
    main()
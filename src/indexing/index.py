import sys
from pathlib import Path
import json
import os

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

def load_checkpoint(checkpoint_path):
    """Load checkpoint to resume indexing"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        print(f"✓ Loaded checkpoint: {checkpoint['processed_batches']}/{checkpoint['total_batches']} batches processed")
        return checkpoint
    return None

def save_checkpoint(checkpoint_path, processed_batches, total_batches, last_index):
    """Save checkpoint after each batch"""
    checkpoint = {
        'processed_batches': processed_batches,
        'total_batches': total_batches,
        'last_index': last_index
    }
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f)

def delete_checkpoint(checkpoint_path):
    """Delete checkpoint file after completion"""
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("✓ Checkpoint file deleted (indexing complete)")

def main():
    record_type = "word"
    word = Word(file_path=f"{PROJECT_ROOT}/dataset/words/[JA-JA]_大辞泉_第二版.jsonl")
    word_loader = word.init_loader()
    data = word_loader.load()

    # 1. Data Preparation for Retrieval
    text_splitter = CharacterTextSplitter(chunk_size=4000)
    chunks = text_splitter.split_documents(data)

    # Initialize Text Embedding with cache
    embeddings = ONNXEmbeddings(
        model_path=f"{PROJECT_ROOT}/model/gte-multilingual-base_onnx",
        cache_size=1000
    )

    # 2. Indexing with optimized vector database settings
    persist_directory = f"{PROJECT_ROOT}/vector_database/words"
    checkpoint_path = f"{PROJECT_ROOT}/checkpoints/indexing_checkpoint.json"

    # Create checkpoint directory if not exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    collection_metadata = {
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 100,
        "hnsw:M": 16,
    }

    # Load checkpoint if exists
    checkpoint = load_checkpoint(checkpoint_path)
    start_batch = 0
    if checkpoint:
        start_batch = checkpoint['processed_batches']
        print(f"Resuming from batch {start_batch}")
    else:
        print("Starting fresh indexing")

    # Create vector database with batching
    # ChromaDB max batch size is 5461, use a safe value below that
    batch_size = 5000
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    try:
        for batch_idx in tqdm(range(start_batch, total_batches),
                             desc=f"Indexing {record_type}",
                             initial=start_batch,
                             total=total_batches):
            i = batch_idx * batch_size
            batch = chunks[i : i + batch_size]

            if i == 0 and start_batch == 0:
                # Create new database
                _ = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    collection_name=f"{record_type}_vector",
                    persist_directory=persist_directory,
                    collection_metadata=collection_metadata
                )
            else:
                # Add to existing database
                vector_db = Chroma(
                    persist_directory=persist_directory,
                    collection_name=f"{record_type}_vector",
                    embedding_function=embeddings
                )
                vector_db.add_documents(batch)

            # Save checkpoint after each batch
            save_checkpoint(checkpoint_path, batch_idx + 1, total_batches, i + len(batch))

        # Delete checkpoint after successful completion
        delete_checkpoint(checkpoint_path)
        print(f"✓ Indexing complete! Total chunks indexed: {len(chunks)}")

    except KeyboardInterrupt:
        print(f"\n⚠ Indexing interrupted! Checkpoint saved at batch {batch_idx}")
        print(f"Run the script again to resume from this point.")
        raise
    except Exception as e:
        print(f"\n✗ Error during indexing: {e}")
        print(f"Checkpoint saved at batch {batch_idx}")
        print(f"Run the script again to resume from this point.")
        raise

if __name__ == "__main__":
    main()
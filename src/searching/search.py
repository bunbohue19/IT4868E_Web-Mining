import sys
import time
import chromadb
from pathlib import Path

import chromadb
import streamlit as st
from langchain_chroma import Chroma

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedding.embedding_model import ONNXEmbeddings


def main(model_path=f"{PROJECT_ROOT}/model/gte-multilingual-base-JA_onnx"):

    # 1. Initialize Text Embedding with cache
    embeddings = ONNXEmbeddings(
        model_path=model_path,
        cache_size=1000 
    )

    # 2. Connect to vector database
    persist_directory = f"{PROJECT_ROOT}/vector_database/words"
    persistent_client = chromadb.PersistentClient(path=persist_directory)
    return Chroma(
        client=persistent_client,
        collection_name="word_vector",
        embedding_function=embeddings,
    )
    # 3. Run example queries with latency tracking

    test_queries = [
        ("Bố", 1),
        ("母", 3),
        ("食べる", 5),
        ("Bố", 1), 
    ]

    total_latencies = []

    for i, (query, k) in enumerate(test_queries, 1):
        start_time = time.perf_counter()
        results = lookup(vector_store, query, k=k)
        latency = (time.perf_counter() - start_time) * 1000
        total_latencies.append(latency)

        is_repeat = any(q == query for q, _ in test_queries[:i-1])
        cache_indicator = "CACHE" if is_repeat else "COLD"

        status = "✅" if latency < 200 else "⚠️"

        print(f"\nQuery {i}: '{query}' (k={k})")
        print(f"  {cache_indicator} | Latency: {latency:.2f}ms {status}")
        print(f"  Results: {len(results)} documents")

        if results:
            preview = results[0].page_content[:100].replace('\n', ' ')
            print(f"  Preview: {preview}...")

    # Summary
    print("\n" + "="*80)
    print("Performance Summary")
    print("="*80)

    avg_latency = sum(total_latencies) / len(total_latencies)
    max_latency = max(total_latencies)
    min_latency = min(total_latencies)

    print(f"\nLatency Statistics:")
    print(f"  • Average: {avg_latency:.2f}ms")
    print(f"  • Min:     {min_latency:.2f}ms")
    print(f"  • Max:     {max_latency:.2f}ms")
    print(f"  • Target:  <200ms")

    if avg_latency < 200:
        print(f"\n✅ Target achieved! Average latency is {avg_latency:.2f}ms")
    else:
        print(f"\n⚠️  Average latency ({avg_latency:.2f}ms) exceeds target")

if __name__ == "__main__":
    main()
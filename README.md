# Japanese Daijisen Dictionary Vector Search

An end-to-end pipeline that exports a multilingual embedding model to ONNX, indexes a Japanese dictionary into a Chroma vector store, and serves a Streamlit UI to search for relevant entries.

## Project Layout
- `dataset/words/[JA-JA]_大辞泉_第二版.jsonl` – source dictionary in JSONL.
- `model/gte-multilingual-base-JA_onnx/` – exported ONNX model + tokenizer.
- `vector_database/words/` – persisted Chroma collection (`word_vector`) created by indexing.
- `src/indexing/index.py` – chunks the dataset, embeds with ONNX, and writes to Chroma.
- `src/searching/search.py` – Streamlit app for querying the vector DB.
- `src/export_onnx/export.py` – exports the Hugging Face model to ONNX.
- `src/embedding/embedding_model.py` – ONNX embedding wrapper.

## Quickstart
1) Install dependencies (Python 3.10+ recommended):
   ```bash
   pip install streamlit chromadb langchain langchain-chroma langchain-community \
       transformers onnxruntime optimum torch tqdm
   ```

2) Export the embedding model to ONNX (requires a Hugging Face token):
   ```bash
   export HF_TOKEN="hf_..."                       # your token
   export HF_HOME="$(pwd)/model"                  # or another directory
   python src/export_onnx/export.py
   ```
   The export writes to `model/gte-multilingual-base-JA_onnx/`.

3) Build the vector database:
   ```bash
   python src/indexing/index.py
   ```
   This creates/updates `vector_database/words` with the `word_vector` collection.

4) Run the search UI:
   ```bash
   streamlit run src/searching/search.py
   ```
   Enter a query and adjust the result count; matching dictionary entries and their formatted metadata will be shown.

## Notes
- If `vector_database/words` is missing, (re)run the indexing step.
- The dataset is expected at `dataset/words/[JA-JA]_大辞泉_第二版.jsonl`; adjust `src/indexing/index.py` if your path differs.
- For GPU inference with ONNX Runtime, install the appropriate `onnxruntime-gpu` build and drivers; otherwise CPU is used.

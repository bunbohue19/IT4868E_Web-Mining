import os
import numpy as np
import onnxruntime as ort
from pathlib import Path
from transformers import AutoTokenizer
from typing import List, Union
from functools import lru_cache
import hashlib

class ONNXEmbeddings:
    """ONNX inference for embedding models with caching support"""

    def __init__(self, model_path: Union[str, Path], cache_size: int = 1000):
        """
        Initialize the ONNX model for inference

        Args:
            model_path: Path to directory containing model.onnx and tokenizer files
            cache_size: Maximum number of cached query embeddings (default: 1000)
        """
        self.model_path = Path(model_path).resolve()
        self.cache_size = cache_size
        self._embedding_cache = {}

        # Load tokenizer with absolute path to avoid validation issues
        print(f"Loading tokenizer from: {self.model_path}")

        # Use absolute path and ensure it exists
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")

        # Work around HuggingFace validation issue with paths containing spaces
        # by temporarily changing directory
        original_dir = os.getcwd()
        try:
            os.chdir(self.model_path.parent)
            relative_path = self.model_path.name
            self.tokenizer = AutoTokenizer.from_pretrained(
                relative_path,
                trust_remote_code=True,
                local_files_only=True
            )
        finally:
            os.chdir(original_dir)
        
        # Load ONNX model
        onnx_model_path = self.model_path / "model.onnx"
        print(f"Loading ONNX model from: {onnx_model_path}")
        
        # Create inference session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3  # Suppress warnings
        
        # Use available providers (CPU or GPU)
        providers = ['CPUExecutionProvider']
        if ort.get_device() == 'GPU':
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(
            str(onnx_model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        # Get output names
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"Model loaded with providers: {self.session.get_providers()}")
        print(f"Output names: {self.output_names}")
        
    def mean_pooling(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Apply mean pooling to get sentence embeddings
        
        Args:
            token_embeddings: Token-level embeddings [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Sentence embeddings [batch_size, hidden_dim]
        """
        # Expand attention mask to match embedding dimensions
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        input_mask_expanded = np.broadcast_to(
            input_mask_expanded, 
            token_embeddings.shape
        )
        
        # Sum embeddings with masking
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        
        # Sum mask values
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        
        # Calculate mean
        return sum_embeddings / sum_mask
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length

        Args:
            embeddings: Input embeddings [batch_size, hidden_dim]

        Returns:
            Normalized embeddings [batch_size, hidden_dim]
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        return embeddings / norms

    def _get_cache_key(self, text: str, normalize: bool, max_length: int) -> str:
        """Generate cache key for a query"""
        key_string = f"{text}|{normalize}|{max_length}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Union[np.ndarray, None]:
        """Retrieve embedding from cache"""
        return self._embedding_cache.get(cache_key)

    def _add_to_cache(self, cache_key: str, embedding: np.ndarray):
        """Add embedding to cache with LRU eviction"""
        if len(self._embedding_cache) >= self.cache_size:
            # Simple LRU: remove oldest entry
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
        self._embedding_cache[cache_key] = embedding

    def clear_cache(self):
        """Clear the embedding cache"""
        self._embedding_cache.clear()
    
    def encode(self, sentences: Union[str, List[str]], batch_size: int = 32, normalize: bool = True, max_length: int = 128, show_progress: bool = False, use_cache: bool = True) -> np.ndarray:
        """
        Encode sentences into embeddings with caching support

        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            max_length: Maximum sequence length
            show_progress: Whether to show progress bar
            use_cache: Whether to use embedding cache (default: True)

        Returns:
            Embeddings as numpy array [num_sentences, hidden_dim]
        """
        # Handle single sentence
        is_single = isinstance(sentences, str)
        if is_single:
            sentences = [sentences]

        # Check cache for single queries (most common use case)
        if use_cache and len(sentences) == 1:
            cache_key = self._get_cache_key(sentences[0], normalize, max_length)
            cached_embedding = self._get_from_cache(cache_key)
            if cached_embedding is not None:
                return cached_embedding

        all_embeddings = []
        num_batches = (len(sentences) + batch_size - 1) // batch_size
        
        # Process in batches
        for batch_idx in range(0, len(sentences), batch_size):
            batch_sentences = sentences[batch_idx:batch_idx + batch_size]
            
            if show_progress:
                current_batch = batch_idx // batch_size + 1
                print(f"Processing batch {current_batch}/{num_batches}...", end='\r')
            
            # Tokenize
            encoded = self.tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np"
            )
            
            # Prepare inputs for ONNX
            onnx_inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64)
            }
            
            # Run inference
            outputs = self.session.run(self.output_names, onnx_inputs)
            
            # outputs[0] is last_hidden_state: [batch_size, seq_len, hidden_dim]
            token_embeddings = outputs[0]
            
            # Apply mean pooling
            embeddings = self.mean_pooling(
                token_embeddings, 
                encoded["attention_mask"]
            )
            
            # Normalize if requested
            if normalize:
                embeddings = self.normalize_embeddings(embeddings)
            
            all_embeddings.append(embeddings)
        
        if show_progress:
            print()  # New line after progress

        # Concatenate all batches
        result = np.vstack(all_embeddings)

        # Cache single query results
        if use_cache and len(sentences) == 1:
            cache_key = self._get_cache_key(sentences[0], normalize, max_length)
            self._add_to_cache(cache_key, result)

        return result
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.encode(texts, show_progress=False).tolist()
            
    def embed_query(self, query: str) -> List[float]:
        return self.encode([query], show_progress=False)[0].tolist()
    
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings
        
        Args:
            embeddings1: First set of embeddings [n, hidden_dim]
            embeddings2: Second set of embeddings [m, hidden_dim]
        
        Returns:
            Similarity matrix [n, m]
        """
        return np.dot(embeddings1, embeddings2.T)
    
    def find_most_similar(self, query: Union[str, np.ndarray], candidates: Union[List[str], np.ndarray], top_k: int = 5) -> List[tuple]:
        """
        Find most similar candidates to a query
        
        Args:
            query: Query sentence or embedding
            candidates: List of candidate sentences or embeddings
            top_k: Number of top results to return
        
        Returns:
            List of (index, similarity_score) tuples
        """
        # Encode if needed
        if isinstance(query, str):
            query_emb = self.encode(query)
        else:
            query_emb = query.reshape(1, -1) if query.ndim == 1 else query
        
        if isinstance(candidates, list):
            candidate_embs = self.encode(candidates)
        else:
            candidate_embs = candidates
        
        # Compute similarities
        similarities = self.similarity(query_emb, candidate_embs)[0]
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]


# if __name__ == "__main__":
#     # Initialize model
#     model_path = "/Users/locseo/Desktop/Master DS/IT4868E_Web mining/model/gte-multilingual-base-JA_onnx"
#     model = ONNXEmbeddingModel(model_path)
    
#     print("\nTest 1: Single Sentence Encoding")
    
#     # Encode single sentence
#     text = "Hello world"
#     embedding = model.encode(text)
#     print(f"Text: '{text}'")
#     print(f"Embedding shape: {embedding.shape}")
#     print(f"First 5 values: {embedding[0][:5]}")
#     print(f"L2 norm: {np.linalg.norm(embedding[0]):.6f}")
    
#     print("\nTest 2: Multiple Sentences & Similarity")
    
#     # Encode multiple sentences
#     sentences = [
#         "First sentence", 
#         "Second sentence", 
#         "Third sentence",
#         "The first statement",
#         "Another sentence here"
#     ]
    
#     embeddings = model.encode(sentences, batch_size=32, normalize=True)
#     print(f"Encoded {len(sentences)} sentences")
#     print(f"Embeddings shape: {embeddings.shape}")
    
#     # Compute similarity matrix
#     similarity_matrix = model.similarity(embeddings, embeddings)
#     print(f"\nSimilarity matrix shape: {similarity_matrix.shape}")
#     print("\nPairwise similarities:")
#     for i in range(len(sentences)):
#         for j in range(i+1, len(sentences)):
#             sim = similarity_matrix[i, j]
#             print(f"  '{sentences[i]}' <-> '{sentences[j]}': {sim:.4f}")
    
#     print("\nTest 3: Find Most Similar")
    
#     query = "First sentence"
#     candidates = sentences[1:]  # Exclude the query itself
    
#     print(f"Query: '{query}'")
#     print(f"\nTop 3 most similar candidates:")
    
#     results = model.find_most_similar(query, candidates, top_k=3)
#     for rank, (idx, score) in enumerate(results, 1):
#         print(f"  {rank}. '{candidates[idx]}' (score: {score:.4f})")
    
#     print("\nTest 4: Multilingual Support")
    
#     multilingual_texts = [
#         "Machine learning is fascinating",
#         "機械学習は魅力的です",  # Japanese: Machine learning is fascinating
#         "L'apprentissage automatique est fascinant",  # French
#     ]
    
#     ml_embeddings = model.encode(multilingual_texts, normalize=True)
#     ml_similarities = model.similarity(ml_embeddings, ml_embeddings)
    
#     print("Cross-lingual similarities:")
#     for i, text1 in enumerate(multilingual_texts):
#         for j, text2 in enumerate(multilingual_texts):
#             if i < j:
#                 sim = ml_similarities[i, j]
#                 print(f"  {sim:.4f}: '{text1[:30]}...' <-> '{text2[:30]}...'")

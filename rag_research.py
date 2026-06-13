"""
Advanced Retrieval-Augmented Generation (RAG) for Legal Complaint Analysis
Research-grade retriever with:
- Configurable embedding and reranking models
- Retrieval metrics and scoring
- Context quality assessment
- Dynamic threshold adjustment
- Comprehensive error handling
"""

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
import time
from datetime import datetime

from config import RAGConfig
from logger_config import setup_logging, ContextualTimer, MetricsTracker

logger = setup_logging(RAGConfig.LOG_DIR, RAGConfig.LOG_LEVEL)


class FAISSRetrieverResearch:
    """
    Advanced FAISS-based retriever with research-grade metrics and configuration.
    
    Improvements over original:
    - Configurable embedding model selection
    - Flexible reranking with score thresholding
    - Detailed retrieval metrics tracking
    - Context quality scoring
    - Caching support
    - Comprehensive error handling
    """
    
    def __init__(self,
                 embedding_model_name: str = None,
                 reranker_model_name: str = None,
                 index_path: str = None,
                 initial_k: int = None,
                 final_k: int = None,
                 reranking_threshold: float = None,
                 use_cache: bool = None,
                 cache_dir: str = None):
        """
        Initialize advanced retriever.
        
        Args:
            embedding_model_name: Embedding model (uses config default if None)
            reranker_model_name: Cross-encoder model (uses config default if None)
            index_path: Path to FAISS index (uses config default if None)
            initial_k: Candidates to retrieve before reranking
            final_k: Final results after reranking
            reranking_threshold: Score threshold for including results
            use_cache: Enable query caching
            cache_dir: Cache directory
        """
        self.config = RAGConfig.get_config()
        
        self.embedding_model_name = embedding_model_name or self.config.get('EMBEDDING_MODEL', RAGConfig.EMBEDDING_MODEL)
        self.reranker_model_name = reranker_model_name or self.config.get('RERANKER_MODEL', RAGConfig.RERANKER_MODEL)
        self.index_path = index_path or self.config.get('INDEX_PATH', RAGConfig.INDEX_PATH)
        self.initial_k = initial_k or self.config.get('INITIAL_RETRIEVE_K', RAGConfig.INITIAL_RETRIEVE_K)
        self.final_k = final_k or self.config.get('FINAL_RETRIEVE_K', RAGConfig.FINAL_RETRIEVE_K)
        self.reranking_threshold = reranking_threshold or self.config.get('RERANKING_THRESHOLD', RAGConfig.RERANKING_THRESHOLD)
        
        self.use_cache = use_cache if use_cache is not None else self.config.get('USE_QUERY_CACHE', RAGConfig.USE_QUERY_CACHE)
        self.cache_dir = Path(cache_dir or self.config.get('QUERY_CACHE_DIR', RAGConfig.QUERY_CACHE_DIR))
        
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker(self.config.get('METRICS_DIR', RAGConfig.METRICS_DIR))
        
        logger.info(f"Initializing FAISSRetrieverResearch",
                   extra={'extra_fields': {
                       'embedding_model': self.embedding_model_name,
                       'reranker_model': self.reranker_model_name,
                       'initial_k': self.initial_k,
                       'final_k': self.final_k,
                       'use_cache': self.use_cache
                   }})
        
        # Load models
        self._load_models()
        
        # Load FAISS index
        self._load_index()
    
    def _load_models(self):
        """Load embedding and reranking models."""
        with ContextualTimer(logger, f"Loading embedding model: {self.embedding_model_name}"):
            self.model = SentenceTransformer(self.embedding_model_name)
        
        with ContextualTimer(logger, f"Loading reranker model: {self.reranker_model_name}"):
            self.reranker = CrossEncoder(self.reranker_model_name)
        
        logger.info("Models loaded successfully")
    
    def _load_index(self):
        """Load FAISS index and metadata."""
        try:
            if not Path(self.index_path).exists():
                logger.error(f"FAISS index not found: {self.index_path}")
                raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
            
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            
            self.index = data["index"]
            self.metadatas = data["metadatas"]
            self.texts = data["texts"]
            self.embedding_model = data.get("embedding_model", "unknown")
            self.index_type = data.get("index_type", "unknown")
            
            logger.info(f"Loaded FAISS index from {self.index_path}",
                       extra={'extra_fields': {
                           'total_vectors': self.index.ntotal,
                           'embedding_model': self.embedding_model,
                           'index_type': self.index_type
                       }})
        
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {str(e)}")
            raise
    
    def _get_cache_key(self, query: str, k: int) -> str:
        """Generate cache key for query."""
        import hashlib
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{query_hash}_{k}.pkl"
    
    def _get_cached_result(self, query: str, k: int) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached result if exists and not expired."""
        if not self.use_cache:
            return None
        
        cache_file = self.cache_dir / self._get_cache_key(query, k)
        
        if not cache_file.exists():
            return None
        
        # Check TTL
        cache_age = time.time() - cache_file.stat().st_mtime
        ttl = self.config.get('QUERY_CACHE_TTL', RAGConfig.QUERY_CACHE_TTL)
        
        if cache_age > ttl:
            cache_file.unlink()  # Remove expired cache
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {str(e)}")
            return None
    
    def _save_cached_result(self, query: str, k: int, result: List[Dict[str, Any]]):
        """Save result to cache."""
        if not self.use_cache:
            return
        
        try:
            cache_file = self.cache_dir / self._get_cache_key(query, k)
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {str(e)}")
    
    def retrieve(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve and rerank top-k results for query.
        
        Two-stage process:
        1. Dense retrieval: Use embedding model for initial candidates
        2. Reranking: Use cross-encoder to score and rerank
        
        Args:
            query: Legal complaint or query text
            k: Number of results to return (uses config default if None)
        
        Returns:
            List of {"content": text, "metadata": dict, "scores": dict} dicts
        """
        k = k or self.final_k
        
        # Check cache
        cached_result = self._get_cached_result(query, k)
        if cached_result:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return cached_result
        
        retrieval_start = time.time()
        
        try:
            # Stage 1: Dense retrieval
            query_emb = self.model.encode([query])
            
            # Normalize embeddings for cosine similarity
            query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
            
            # Search FAISS
            scores_initial, indices = self.index.search(
                np.array(query_emb, dtype=np.float32),
                self.initial_k
            )
            
            # Collect candidates with metadata
            candidates = []
            candidate_texts = []
            retrieval_scores = []
            
            for idx, score in zip(indices[0], scores_initial[0]):
                if idx >= len(self.texts):
                    continue
                
                candidates.append((self.texts[idx], self.metadatas[idx]))
                candidate_texts.append(self.texts[idx])
                retrieval_scores.append(float(score))
            
            logger.debug(f"Stage 1 (dense retrieval) found {len(candidates)} candidates")
            
            # Stage 2: Reranking
            if candidates:
                reranked_results = self.reranker.rank(
                    query,
                    candidate_texts,
                    top_k=min(len(candidates), max(k * 3, len(candidates)))  # Get more for threshold filtering
                )
                
                # Apply threshold and create results
                final_results = []
                for ranked in reranked_results:
                    score = float(ranked['score'])
                    
                    # Always include results if threshold is 0, otherwise apply threshold
                    if self.reranking_threshold == 0 or score >= self.reranking_threshold:
                        corpus_id = ranked['corpus_id']
                        text, metadata = candidates[corpus_id]
                        
                        # Update metadata with scores
                        metadata_copy = metadata.copy()
                        metadata_copy['retrieval_score'] = retrieval_scores[corpus_id]
                        metadata_copy['reranking_score'] = score
                        metadata_copy['chunk_rank'] = len(final_results) + 1
                        
                        final_results.append({
                            "content": text,
                            "metadata": metadata_copy,
                            "scores": {
                                "retrieval": retrieval_scores[corpus_id],
                                "reranking": score,
                                "combined": (retrieval_scores[corpus_id] + score) / 2
                            }
                        })
                        
                        if len(final_results) >= k:
                            break
                
                reranking_scores = [r['scores']['reranking'] for r in final_results]
                logger.debug(f"Stage 2 (reranking) returned {len(final_results)} results "
                           f"(threshold: {self.reranking_threshold})")
            else:
                final_results = []
                reranking_scores = []
            
            # Track metrics
            retrieval_time = time.time() - retrieval_start
            self.metrics_tracker.record_retrieval(
                query,
                k,
                retrieval_time,
                retrieval_scores,
                reranking_scores,
                len(final_results)
            )
            
            logger.info(f"Retrieval complete for query: {query[:50]}...",
                       extra={'extra_fields': {
                           'query_length': len(query),
                           'retrieval_time_ms': retrieval_time * 1000,
                           'results_returned': len(final_results),
                           'threshold_applied': self.reranking_threshold
                       }})
            
            # Cache result
            self._save_cached_result(query, k, final_results)
            
            return final_results
        
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}", exc_info=True)
            raise
    
    def retrieve_with_diversity(self, query: str, k: int = None,
                               diversity_penalty: float = 0.1) -> List[Dict[str, Any]]:
        """
        Retrieve results with diversity penalty to avoid redundant documents.
        
        Useful for legal documents where same section might appear in multiple acts.
        
        Args:
            query: Query text
            k: Number of results
            diversity_penalty: Penalty for similar act names (0-1)
        
        Returns:
            Diversified results
        """
        k = k or self.final_k
        
        # Get base results
        base_results = self.retrieve(query, k * 2)  # Get more to filter
        
        # Apply diversity filtering
        selected = []
        selected_acts = set()
        
        for result in base_results:
            act_name = result['metadata'].get('act_name', 'unknown')
            
            if act_name not in selected_acts:
                selected.append(result)
                selected_acts.add(act_name)
                
                if len(selected) >= k:
                    break
            else:
                # Penalize score for same act
                result['scores']['combined'] *= (1 - diversity_penalty)
        
        logger.debug(f"Diversity filtering: {len(base_results)} -> {len(selected)} results")
        
        return selected[:k]


def get_merged_context(query: str, k: int = None,
                      retriever: FAISSRetrieverResearch = None) -> List[str]:
    """
    High-level function to get merged contexts from retrieval.
    
    Args:
        query: Legal complaint query
        k: Number of results
        retriever: Retriever instance (creates new if None)
    
    Returns:
        List of context strings
    """
    if retriever is None:
        retriever = FAISSRetrieverResearch()
    
    k = k or RAGConfig.FINAL_RETRIEVE_K
    
    results = retriever.retrieve(query, k)
    
    # Extract and format contexts
    contexts = []
    for result in results:
        metadata = result['metadata']
        
        # Format context with source information
        source_info = f"[{metadata.get('act_name', 'Unknown')} - "
        if metadata.get('section_number'):
            source_info += f"{metadata.get('section_number')} - "
        source_info += f"Page {metadata.get('page_number', '?')}]"
        
        formatted_context = f"{source_info}\n{result['content']}"
        contexts.append(formatted_context)
    
    return contexts


# Global retriever instance (lazy loaded)
_retriever_instance = None


def get_retriever() -> FAISSRetrieverResearch:
    """Get or create global retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = FAISSRetrieverResearch()
    return _retriever_instance


if __name__ == "__main__":
    """
    Test retriever functionality.
    """
    import json
    
    try:
        # Initialize retriever
        retriever = FAISSRetrieverResearch()
        
        # Test queries
        test_queries = [
            "What is the punishment for theft?",
            "Copyright infringement penalties",
            "Fundamental rights under constitution"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print('='*60)
            
            results = retriever.retrieve(query, k=5)
            
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"Act: {result['metadata'].get('act_name')}")
                print(f"Section: {result['metadata'].get('section_number')}")
                print(f"Scores: Retrieval={result['scores']['retrieval']:.3f}, "
                      f"Reranking={result['scores']['reranking']:.3f}")
                print(f"Content: {result['content'][:200]}...")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

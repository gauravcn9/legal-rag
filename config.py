"""
Configuration management for Legal RAG Research Pipeline
Centralizes all tunable parameters to ensure reproducibility and ease of experimentation
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
import yaml

# Get workspace root dynamically
WORKSPACE_ROOT = Path(__file__).parent


class RAGConfig:
    """
    Centralized configuration for RAG pipeline.
    Load from config.yaml or environment variables with defaults.
    """

    # =======================
    # PDF Processing Configuration
    # =======================
    
    # Chunking strategy optimized for legal documents
    # Optimal range for unstructured legal PDFs: 600-800 tokens
    # Overlap: 10-15% (60-120 for 600-800 chunk size)
    CHUNK_SIZE: int = 700  # tokens
    CHUNK_OVERLAP: int = 85  # ~12.1% overlap
    
    # Legal-specific separators in priority order
    # Preserve legal section boundaries
    SEPARATORS: list = [
        "\n\nSection ",      # Act sections
        "\n\nArticle ",      # Constitution articles
        "\n\nClause ",       # Contract clauses
        "\n\nRule ",         # Rules
        "\n\nSchedule ",     # Schedules
        "\n\n",              # Paragraph breaks
        "\n",                # Line breaks
        " ",                 # Word level
        ""                   # Character level (fallback)
    ]
    
    # Maximum workers for parallel processing
    MAX_WORKERS: int = 4
    
    # =======================
    # Embedding Configuration
    # =======================
    
    # Legal-domain optimized embedding models
    # Primary: UAE-Large-V1 (1024 dims, legal-specific)
    # Fallback: sentence-transformers/legal-NLI-MiniLM-L6-v1
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"  # Using fallback for availability
    
    # Embedding cache to avoid recomputation
    USE_EMBEDDING_CACHE: bool = True
    EMBEDDING_CACHE_DIR: str = str(WORKSPACE_ROOT / "embeddings_cache")
    
    # =======================
    # FAISS Index Configuration
    # =======================
    
    # Index type: "flat_l2" (current), "flat_ip" (normalized, better for embeddings), "hnsw" (hierarchical)
    INDEX_TYPE: str = "flat_ip"  # Inner product on normalized vectors
    
    # HNSW parameters (if using hierarchical indexing)
    HNSW_M: int = 32  # number of bidirectional links
    HNSW_EF_CONSTRUCTION: int = 200
    HNSW_EF_SEARCH: int = 200
    
    # Index persistence
    INDEX_PATH: str = str(WORKSPACE_ROOT / "faiss_index.pkl")
    INDEX_BACKUP_PATH: str = str(WORKSPACE_ROOT / "faiss_index_backup.pkl")
    
    # =======================
    # Retrieval Configuration
    # =======================
    
    # Number of candidates to retrieve before re-ranking
    INITIAL_RETRIEVE_K: int = 20  # Retrieve extra for re-ranking
    
    # Number of final results after re-ranking
    FINAL_RETRIEVE_K: int = 10
    
    # Re-ranker model
    # Current: cross-encoder/ms-marco-MiniLM-L-6-v2 (web search domain)
    # Better: cross-encoder/ms-marco-MiniLM-L-12-v2 (larger, more accurate)
    # Future: legal-domain specific cross-encoder (when available)
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Re-ranking score threshold (0-1)
    # Filters out low-confidence matches
    RERANKING_THRESHOLD: float = 0.0  # Set 0 to include all, adjust based on evaluation
    
    # =======================
    # LLM Configuration
    # =======================
    
    # Model selection (flexible for experimentation)
    LLM_MODEL: str = "deepseek-ai/deepseek-v3.1"
    
    # Model parameters
    LLM_TEMPERATURE: float = 0.0  # 0 for deterministic, 0.3-0.7 for reasoning
    LLM_TOP_P: float = 0.7
    LLM_MAX_TOKENS: int = 8192
    
    # API configuration
    LLM_API_KEY_ENV: str = "OPENAI_API_KEY"
    LLM_API_BASE_URL_ENV: str = "OPENAI_BASE_URL"
    
    # Request timeout (seconds)
    LLM_TIMEOUT: int = 120
    
    # Retry configuration
    LLM_MAX_RETRIES: int = 3
    LLM_RETRY_BACKOFF: float = 2.0  # exponential backoff multiplier
    
    # =======================
    # Prompt Configuration
    # =======================
    
    # System prompt for legal analysis
    SYSTEM_PROMPT: str = """You are an advanced, conservative AI legal reviewer and legal assistant specializing in all domains of Indian law, including but not limited to:
- Criminal laws (e.g., BNS, BNSS, BSA or equivalents)
- Constitutional law (e.g., Fundamental Rights like Articles 14-32)
- Civil laws (e.g., contracts, property, family courts, tribunals, arbitration)
- Corporate/commercial laws (e.g., companies, insolvency, securities regulations)
- Special acts (e.g., SC/ST prevention of atrocities, POCSO, POSH, NDPS)
- Intellectual property laws (e.g., copyrights, patents, trademarks)
- Taxation laws (e.g., income tax, GST)
- Information technology laws (e.g., IT Act or similar)
- Labor laws (e.g., industrial disputes, wages, provident funds)
- And any other relevant Indian acts or statutes strictly based on the complaint and contexts."""
    
    # Version for prompt tracking
    PROMPT_VERSION: str = "1.0"
    
    # =======================
    # Document Paths
    # =======================
    
    # Document directory (parameterized, not hardcoded)
    PDF_DIRECTORY: str = str(WORKSPACE_ROOT.parent / "Documents")
    
    # =======================
    # Logging Configuration
    # =======================
    
    # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_LEVEL: str = "INFO"
    
    # Log directory
    LOG_DIR: str = str(WORKSPACE_ROOT / "logs")
    
    # Log format
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Log file rotation
    LOG_MAX_BYTES: int = 10_485_760  # 10 MB
    LOG_BACKUP_COUNT: int = 5
    
    # =======================
    # Caching Configuration
    # =======================
    
    # Enable query result caching
    USE_QUERY_CACHE: bool = True
    QUERY_CACHE_DIR: str = str(WORKSPACE_ROOT / "query_cache")
    QUERY_CACHE_TTL: int = 3600  # seconds (1 hour)
    
    # =======================
    # Evaluation Configuration
    # =======================
    
    # Metrics to track
    TRACK_METRICS: bool = True
    METRICS_DIR: str = str(WORKSPACE_ROOT / "metrics")
    
    # Save query-response pairs for review
    SAVE_AUDIT_LOG: bool = True
    AUDIT_LOG_DIR: str = str(WORKSPACE_ROOT / "audit_logs")
    
    # =======================
    # Validation Configuration
    # =======================
    
    # Validate JSON output from LLM
    VALIDATE_JSON_OUTPUT: bool = True
    
    # Minimum confidence for charge relevance
    MIN_CHARGE_CONFIDENCE: float = 0.5
    
    # =======================
    # Class Methods
    # =======================
    
    @classmethod
    def load_from_yaml(cls, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return config_data or {}
    
    @classmethod
    def from_env_overrides(cls) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        overrides = {}
        
        # Check for environment variable overrides
        env_vars = {
            'CHUNK_SIZE': int,
            'CHUNK_OVERLAP': int,
            'EMBEDDING_MODEL': str,
            'LLM_MODEL': str,
            'LLM_TEMPERATURE': float,
            'PDF_DIRECTORY': str,
            'LOG_LEVEL': str,
        }
        
        for env_var, type_converter in env_vars.items():
            env_value = os.getenv(f"RAG_{env_var}")
            if env_value:
                try:
                    overrides[env_var] = type_converter(env_value)
                except ValueError:
                    print(f"Warning: Could not convert {env_var}={env_value} to {type_converter}")
        
        return overrides
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """
        Get final configuration by merging:
        1. Class defaults
        2. YAML config file (if exists)
        3. Environment variable overrides
        """
        config = {attr: getattr(cls, attr) for attr in dir(cls) 
                  if not attr.startswith('_') and attr.isupper()}
        
        # Try to load from YAML
        config_yaml_path = WORKSPACE_ROOT / "config.yaml"
        if config_yaml_path.exists():
            yaml_config = cls.load_from_yaml(str(config_yaml_path))
            config.update(yaml_config)
        
        # Apply environment overrides
        env_overrides = cls.from_env_overrides()
        config.update(env_overrides)
        
        return config


# Create directories if they don't exist
def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        RAGConfig.EMBEDDING_CACHE_DIR,
        RAGConfig.LOG_DIR,
        RAGConfig.QUERY_CACHE_DIR,
        RAGConfig.METRICS_DIR,
        RAGConfig.AUDIT_LOG_DIR,
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    config = RAGConfig.get_config()
    print("Configuration loaded successfully:")
    for key, value in sorted(config.items()):
        print(f"  {key}: {value}")

"""
Centralized logging and metrics tracking for Legal RAG Research Pipeline
Provides structured logging, metrics collection, and audit trails for research reproducibility
"""

import logging
import logging.handlers
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import sys


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging compatible with ML pipelines."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        # Add any extra fields
        if hasattr(record, 'extra_fields'):
            log_obj.update(record.extra_fields)
        
        return json.dumps(log_obj)


class MetricsTracker:
    """Track pipeline metrics for performance analysis and research."""
    
    def __init__(self, metrics_dir: str):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def record_retrieval(self, query: str, k: int, retrieval_time: float, 
                        scores: list, reranking_scores: list, top_k_count: int):
        """Record retrieval metrics."""
        metrics_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'query_length': len(query),
            'k_requested': k,
            'retrieval_time_ms': retrieval_time * 1000,
            'mean_retrieval_score': sum(scores) / len(scores) if scores else 0,
            'max_retrieval_score': max(scores) if scores else 0,
            'min_retrieval_score': min(scores) if scores else 0,
            'mean_reranking_score': sum(reranking_scores) / len(reranking_scores) if reranking_scores else 0,
            'max_reranking_score': max(reranking_scores) if reranking_scores else 0,
            'top_k_results': top_k_count,
        }
        
        if 'retrieval' not in self.metrics:
            self.metrics['retrieval'] = []
        self.metrics['retrieval'].append(metrics_entry)
    
    def record_llm_call(self, model: str, query_length: int, context_length: int,
                       llm_time: float, response_length: int, tokens_used: int = None):
        """Record LLM call metrics."""
        metrics_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'model': model,
            'query_length': query_length,
            'context_length': context_length,
            'llm_time_ms': llm_time * 1000,
            'response_length': response_length,
            'tokens_used': tokens_used,
        }
        
        if 'llm' not in self.metrics:
            self.metrics['llm'] = []
        self.metrics['llm'].append(metrics_entry)
    
    def record_pipeline(self, total_time: float, retrieval_time: float, 
                       llm_time: float, contexts_used: int, success: bool, error: str = None):
        """Record end-to-end pipeline metrics."""
        metrics_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_time_ms': total_time * 1000,
            'retrieval_time_ms': retrieval_time * 1000,
            'llm_time_ms': llm_time * 1000,
            'contexts_used': contexts_used,
            'success': success,
            'error': error,
        }
        
        if 'pipeline' not in self.metrics:
            self.metrics['pipeline'] = []
        self.metrics['pipeline'].append(metrics_entry)
    
    def save_metrics(self):
        """Save metrics to file."""
        metrics_file = self.metrics_dir / f"metrics_{self.session_id}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        return str(metrics_file)


class AuditLogger:
    """Log all queries and responses for audit and research purposes."""
    
    def __init__(self, audit_log_dir: str):
        self.audit_dir = Path(audit_log_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.audit_file = self.audit_dir / f"audit_{self.session_id}.jsonl"
    
    def log_query_response(self, query: str, retrieved_contexts: list, 
                          llm_response: str, metadata: Dict[str, Any] = None):
        """
        Log a query-response pair for audit trail.
        Supports research analysis and reproducibility.
        """
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'query': query,
            'retrieved_contexts': retrieved_contexts,
            'llm_response': llm_response,
            'context_count': len(retrieved_contexts),
        }
        
        if metadata:
            audit_entry['metadata'] = metadata
        
        # Append to JSONL file
        with open(self.audit_file, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')
    
    def get_audit_file(self) -> str:
        return str(self.audit_file)


def setup_logging(log_dir: str, log_level: str = "INFO", 
                  use_json: bool = True, log_file_name: str = "rag_pipeline.log") -> logging.Logger:
    """
    Setup centralized logging for the RAG pipeline.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_json: Whether to use JSON format for logs
        log_file_name: Name of the log file
    
    Returns:
        Configured logger instance
    """
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('legal_rag')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler with rotation
    log_file = log_path / log_file_name
    file_handler = logging.handlers.RotatingFileHandler(
        str(log_file),
        maxBytes=10_485_760,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler - only show WARNING and above to reduce noise
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Suppress INFO and DEBUG from console
    
    # Formatter
    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_with_context(logger: logging.Logger, level: str, message: str, 
                    extra_fields: Dict[str, Any] = None):
    """
    Log with additional context fields.
    
    Args:
        logger: Logger instance
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        message: Log message
        extra_fields: Additional fields to include in JSON log
    """
    record = logging.LogRecord(
        name=logger.name,
        level=getattr(logging, level.upper()),
        pathname="",
        lineno=0,
        msg=message,
        args=(),
        exc_info=None
    )
    
    if extra_fields:
        record.extra_fields = extra_fields
    
    getattr(logger, level.lower())(message)


class ContextualTimer:
    """Context manager for timing pipeline operations."""
    
    def __init__(self, logger: logging.Logger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        if exc_type is None:
            self.logger.debug(f"Completed: {self.operation_name} ({self.elapsed:.3f}s)")
        else:
            self.logger.error(
                f"Failed: {self.operation_name} ({self.elapsed:.3f}s)",
                exc_info=(exc_type, exc_val, exc_tb)
            )
        return False


if __name__ == "__main__":
    # Example usage
    logger = setup_logging("./logs", log_level="DEBUG")
    
    metrics = MetricsTracker("./metrics")
    audit = AuditLogger("./audit_logs")
    
    logger.info("Test log message")
    
    with ContextualTimer(logger, "Example operation"):
        time.sleep(0.1)
    
    metrics.record_retrieval("test query", 10, 0.1, [0.9, 0.8, 0.7], [0.85, 0.75], 3)
    metrics.save_metrics()
    
    audit.log_query_response(
        "test query",
        ["context 1", "context 2"],
        '{"charges": []}',
        {"model": "deepseek-v3.1"}
    )
    
    print(f"Metrics saved to: {metrics.metrics_dir}")
    print(f"Audit log saved to: {audit.audit_file}")

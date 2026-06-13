"""
Enhanced Main Interface for Legal RAG System
Research-grade CLI with:
- Comprehensive input validation
- Formatted output with metadata
- Detailed logging and timing
- Error handling and user guidance
- Support for batch processing
- Export capabilities
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_research import LLMInterfaceResearch, process_query, LLMConfigError
from logger_config import setup_logging
from config import RAGConfig

# Setup logging
logger = setup_logging(RAGConfig.LOG_DIR, RAGConfig.LOG_LEVEL)


class LegalRAGCLI:
    """Interactive CLI for Legal RAG System."""
    
    def __init__(self):
        """Initialize CLI."""
        self.llm = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_queries = []
        
        self._initialize()
    
    def _initialize(self):
        """Initialize LLM interface."""
        try:
            self.llm = LLMInterfaceResearch()
            logger.info("CLI initialized successfully")
        except LLMConfigError as e:
            logger.error(f"Configuration error: {str(e)}")
            print(f"\nError: {str(e)}")
            print("Please ensure the following environment variables are set:")
            print(f"  - {RAGConfig.LLM_API_KEY_ENV}")
            print(f"  - {RAGConfig.LLM_API_BASE_URL_ENV}")
            sys.exit(1)
    
    def _print_header(self):
        """Print system header."""
        print("\n" + "="*70)
        print("  LEGAL COMPLAINT ANALYSIS SYSTEM (Research-Grade RAG Pipeline)")
        print("  Powered by: Deep Seek V3.1 + FAISS Vector Database")
        print("="*70)
        print("\nType 'help' for available commands.")
        print("Type 'exit' or 'quit' to end session.\n")
    
    def _format_response(self, response_json_str: str, query: str, response_time: float) -> str:
        """
        Format and display response with metadata.
        
        Args:
            response_json_str: JSON response from LLM
            query: Original query
            response_time: Processing time in seconds
        
        Returns:
            Formatted output string
        """
        output = []
        output.append("\n" + "-"*70)
        output.append("LEGAL ANALYSIS RESULT")
        output.append("-"*70)
        
        try:
            response_data = json.loads(response_json_str)
            
            # Pretty print JSON
            output.append(json.dumps(response_data, indent=2, ensure_ascii=False))
        
        except json.JSONDecodeError:
            output.append("Response (Raw):")
            output.append(response_json_str)
        
        output.append("-"*70)
        output.append(f"Processing Time: {response_time:.2f}s")
        output.append("-"*70 + "\n")
        
        return "\n".join(output)
    
    def _save_session_report(self, export_dir: Optional[str] = None) -> str:
        """
        Save session report with all queries and responses.
        
        Args:
            export_dir: Directory to save report (uses config default if None)
        
        Returns:
            Path to saved report
        """
        if not self.session_queries:
            return None
        
        export_path = Path(export_dir or RAGConfig.AUDIT_LOG_DIR)
        export_path.mkdir(parents=True, exist_ok=True)
        
        report_file = export_path / f"session_report_{self.session_id}.json"
        
        report_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(self.session_queries),
            "queries": self.session_queries
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Session report saved to {report_file}")
        return str(report_file)
    
    def run_interactive(self):
        """Run interactive CLI session."""
        self._print_header()
        
        while True:
            try:
                query = input("\nEnter legal complaint (or 'exit'): ").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.lower() in ['exit', 'quit', 'q']:
                    self._handle_exit()
                    break
                
                elif query.lower() == 'help':
                    self._print_help()
                    continue
                
                elif query.lower() == 'save':
                    report_path = self._save_session_report()
                    if report_path:
                        print(f"\nSession saved to: {report_path}")
                    else:
                        print("\nNo queries to save.")
                    continue
                
                elif query.lower().startswith('config'):
                    self._print_config()
                    continue
                
                # Process query
                self._process_query(query)
            
            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                self._handle_exit()
                break
            
            except Exception as e:
                logger.error(f"Error in CLI loop: {str(e)}", exc_info=True)
                print(f"\nError: {str(e)}")
                print("Please try again or type 'help' for available commands.")
    
    def _process_query(self, query: str):
        """
        Process a single query.
        
        Args:
            query: Legal complaint query
        """
        try:
            print("\nProcessing complaint...")
            
            start_time = time.time()
            response = self.llm.process_query(query)
            response_time = time.time() - start_time
            
            # Display formatted response
            print(self._format_response(response, query, response_time))
            
            # Store in session
            self.session_queries.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "processing_time_seconds": response_time
            })
            
            logger.info(f"Query processed successfully ({response_time:.2f}s)")
        
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}", exc_info=True)
            print(f"\nError processing complaint: {str(e)}")
            print("Please check the logs for details.")
    
    def _print_help(self):
        """Print help message."""
        help_text = """
Available Commands:
  exit/quit/q      - Exit the system and save session
  help             - Show this help message
  save             - Save current session to file
  config           - Show current configuration
  
Quick Tips:
  - Enter your legal complaint and the system will analyze it
  - The system uses RAG (Retrieval-Augmented Generation) to find relevant laws
  - Results are provided in JSON format for easy processing
  - All queries and responses are logged for research purposes
        """
        print(help_text)
    
    def _print_config(self):
        """Print current configuration."""
        config = RAGConfig.get_config()
        
        print("\n" + "="*70)
        print("CURRENT CONFIGURATION")
        print("="*70)
        
        # Categorize config items
        categories = {
            "PDF Processing": [k for k in config if "CHUNK" in k or "SEPARATOR" in k or "WORKER" in k],
            "Embeddings": [k for k in config if "EMBEDDING" in k or "CACHE" in k],
            "Retrieval": [k for k in config if "RETRIEVE" in k or "RERANK" in k],
            "LLM": [k for k in config if "LLM" in k],
            "Logging & Metrics": [k for k in config if "LOG" in k or "METRIC" in k],
            "Paths": [k for k in config if "PATH" in k or "DIR" in k or "DIRECTORY" in k],
        }
        
        for category, keys in categories.items():
            if keys:
                print(f"\n{category}:")
                for key in sorted(keys):
                    value = config.get(key)
                    if key.endswith("URL") and value:
                        value = "***" if len(value) > 20 else value
                    print(f"  {key}: {value}")
        
        print("\n" + "="*70 + "\n")
    
    def _handle_exit(self):
        """Handle system exit."""
        if self.session_queries:
            report_path = self._save_session_report()
            print(f"\nSession saved to: {report_path}")
        
        print("\nThank you for using Legal RAG System.")
        print("Session report and logs saved to: ", RAGConfig.AUDIT_LOG_DIR)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Legal Complaint Analysis System (Research-Grade RAG Pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     # Start interactive mode
  %(prog)s -q "My complaint"   # Process single query
  %(prog)s --config            # Show configuration
        """
    )
    
    parser.add_argument(
        "-q", "--query",
        help="Single query to process (non-interactive mode)"
    )
    
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show current configuration and exit"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file for results (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Show config and exit
    if args.config:
        cli = LegalRAGCLI()
        cli._print_config()
        return
    
    # Single query mode
    if args.query:
        try:
            logger.info(f"Processing single query: {args.query[:100]}...")
            response = process_query(args.query)
            
            # Parse and pretty-print response
            try:
                response_data = json.loads(response)
                output = json.dumps(response_data, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                output = response
            
            print(output)
            
            # Save to file if specified
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(output)
                
                print(f"\nResults saved to: {output_path}")
        
        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            print(f"Error: {str(e)}")
            sys.exit(1)
    
    else:
        # Interactive mode
        cli = LegalRAGCLI()
        cli.run_interactive()


if __name__ == "__main__":
    main()

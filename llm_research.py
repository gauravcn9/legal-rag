"""
Advanced LLM Interface for Legal RAG Pipeline
Research-grade implementation with:
- Configurable model selection and parameters
- JSON response validation and recovery
- Comprehensive prompt management
- Dynamic context optimization
- Error handling and retry logic
- Audit logging
"""

import os
import json
import time
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
import re

from openai import OpenAI, APIError, Timeout
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from config import RAGConfig
from logger_config import setup_logging, ContextualTimer, AuditLogger, MetricsTracker
from rag_research import FAISSRetrieverResearch

load_dotenv()

logger = setup_logging(RAGConfig.LOG_DIR, RAGConfig.LOG_LEVEL)


class State(TypedDict):
    """Pipeline state for LLM RAG workflow."""
    query: str
    contexts: List[str]
    response: str
    metadata: Optional[Dict[str, Any]]


class LLMConfigError(Exception):
    """Configuration error for LLM."""
    pass


class ResponseValidationError(Exception):
    """Response validation error."""
    pass


class LLMInterfaceResearch:
    """
    Research-grade LLM interface with advanced features.
    
    Improvements over original:
    - Configurable model and parameters
    - JSON response validation and recovery
    - Retry logic with exponential backoff
    - Comprehensive error handling
    - Audit logging for all interactions
    - Metrics tracking
    """
    
    def __init__(self,
                 model: str = None,
                 api_key_env: str = None,
                 api_base_url_env: str = None,
                 temperature: float = None,
                 top_p: float = None,
                 max_tokens: int = None,
                 max_retries: int = None,
                 timeout: int = None):
        """
        Initialize LLM interface.
        
        Args:
            model: Model name (uses config default if None)
            api_key_env: Environment variable for API key
            api_base_url_env: Environment variable for API base URL
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens in response
            max_retries: Maximum number of retries on failure
            timeout: Request timeout in seconds
        """
        self.config = RAGConfig.get_config()
        
        self.model = model or self.config.get('LLM_MODEL', RAGConfig.LLM_MODEL)
        self.temperature = temperature if temperature is not None else self.config.get('LLM_TEMPERATURE', RAGConfig.LLM_TEMPERATURE)
        self.top_p = top_p if top_p is not None else self.config.get('LLM_TOP_P', RAGConfig.LLM_TOP_P)
        self.max_tokens = max_tokens or self.config.get('LLM_MAX_TOKENS', RAGConfig.LLM_MAX_TOKENS)
        self.max_retries = max_retries or self.config.get('LLM_MAX_RETRIES', RAGConfig.LLM_MAX_RETRIES)
        self.timeout = timeout or self.config.get('LLM_TIMEOUT', RAGConfig.LLM_TIMEOUT)
        
        api_key_env = api_key_env or self.config.get('LLM_API_KEY_ENV', RAGConfig.LLM_API_KEY_ENV)
        api_base_url_env = api_base_url_env or self.config.get('LLM_API_BASE_URL_ENV', RAGConfig.LLM_API_BASE_URL_ENV)
        
        # Get API credentials
        api_key = os.getenv(api_key_env)
        api_base_url = os.getenv(api_base_url_env)
        
        if not api_key:
            raise LLMConfigError(f"Missing API key. Set {api_key_env} environment variable")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key, base_url=api_base_url, timeout=self.timeout)
        
        # Initialize components
        self.retriever = FAISSRetrieverResearch()
        self.audit_logger = AuditLogger(self.config.get('AUDIT_LOG_DIR', RAGConfig.AUDIT_LOG_DIR))
        self.metrics_tracker = MetricsTracker(self.config.get('METRICS_DIR', RAGConfig.METRICS_DIR))
        
        logger.info(f"Initialized LLMInterfaceResearch",
                   extra={'extra_fields': {
                       'model': self.model,
                       'temperature': self.temperature,
                       'top_p': self.top_p,
                       'max_tokens': self.max_tokens
                   }})
    
    def _generate_system_prompt(self) -> str:
        """Generate system prompt from config."""
        return self.config.get('SYSTEM_PROMPT', RAGConfig.SYSTEM_PROMPT)
    
    def _generate_user_prompt(self, query: str, contexts: List[str]) -> str:
        """Generate user prompt with contexts."""
        prompt = f"""
TASK:
Analyze the given complaint and provided contexts.
Use ONLY the contexts and valid legal sources — never hallucinate or assume laws/sections not supported by contexts.
Select ONLY relevant laws/sections based on the complaint's domain (e.g., criminal, civil, consumer, cyber, family, labor, constitutional violations like fundamental rights, or hybrid). Omit any laws/sections that are completely irrelevant or not applicable.
Do NOT invent Act names or Section numbers that are NOT present in the contexts or in the local statute index.
If the context is insufficient or irrelevant for any aspect, explicitly state in the note: "Insufficient context for [specific aspect, e.g., constitutional violation]."

INPUT COMPLAINT:
{query}

CONTEXTS (for reference only, do not invent beyond these):
{chr(10).join(contexts)}

OUTPUT FORMAT:
Respond ONLY in valid JSON.
Include fields ONLY if applicable. Omit irrelevant fields (e.g., bailable/cognizable/max_imprisonment/penalty for non-criminal laws; applicability if obvious or not applicable; omit 'charges' entirely if no charges apply).

{{
    "victim": "Identified victim, if any",
    "culprit": "Identified culprit, if any",
    "charges": [
        {{
            "law": "Act + Section/Article (e.g., BNS Sec. 115 or Constitution Article 21 or IT Act etc)",
            "description": "Brief description of the offence/violation",
            "bailable": "Yes/No (omit if not criminal)",
            "cognizable": "Yes/No (omit if not criminal)",
            "max_imprisonment": "e.g., Up to 3 years (omit if not applicable)",
            "penalty": "Fine (if mentioned, e.g., Up to Rs. 10,000; omit if not applicable)",
            "applicability": "Individual / Group / Company (omit if obvious or not applicable)"
        }}
    ],
    "note": "Clarifications, limitations, or insufficiency message (e.g., insufficient context, BNSS procedure for FIR if criminal)"
}}

RULES:
- Avoid outdated laws (e.g., IPC/CrPC/IEA). Use ONLY current equivalents (e.g., BNS/BNSS/BSA/IT Act or similar based on contexts).
- If a law/section is unclear or not supported in context, omit it entirely → do not guess or mark it as anything.
- Handle hybrid domains by listing all applicable laws without duplication.
- Do not add unnecessary text outside JSON.
- Prefer brevity, clarity, and correctness.
- In 'note', mention procedures (e.g., BNSS for FIR, tribunal referral) if relevant, but avoid repeating charges. If no charges apply, provide guidance on next steps (e.g., civil suit, consumer forum).
- For non-criminal domains (e.g., civil, labor, taxation), omit criminal-specific fields like bailable/cognizable, and focus on relevant remedies or procedures.
"""
        return prompt
    
    def _validate_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Validate and parse JSON response from LLM.
        
        Handles:
        - Extracting JSON from text with markdown code blocks
        - Recovering from minor JSON formatting issues
        - Comprehensive error reporting
        
        Args:
            response_text: Raw response from LLM
        
        Returns:
            Parsed JSON as dictionary
        
        Raises:
            ResponseValidationError: If JSON is invalid and cannot be recovered
        """
        # Try direct parsing first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON from markdown code blocks
        json_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{.*\}',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # Try to recover common JSON issues
        try:
            # Remove trailing commas
            cleaned = re.sub(r',(\s*[}\]])', r'\1', response_text)
            # Fix common escaping issues
            cleaned = cleaned.replace("\\'", "'").replace('\\"', '"')
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        logger.error(f"Failed to parse JSON response: {response_text[:200]}...")
        raise ResponseValidationError(f"Invalid JSON response from LLM")
    
    def _call_llm_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call LLM with retry logic and exponential backoff.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
        
        Returns:
            LLM response text
        
        Raises:
            APIError: If all retries fail
        """
        backoff_multiplier = self.config.get('LLM_RETRY_BACKOFF', RAGConfig.LLM_RETRY_BACKOFF)
        
        for attempt in range(self.max_retries):
            try:
                with ContextualTimer(logger, f"LLM call (attempt {attempt + 1}/{self.max_retries})"):
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                    )
                
                response_text = completion.choices[0].message.content
                logger.debug(f"LLM response received ({len(response_text)} chars)")
                
                return response_text
            
            except (APIError, Timeout) as e:
                if attempt < self.max_retries - 1:
                    wait_time = (backoff_multiplier ** attempt)
                    logger.warning(f"LLM call failed (attempt {attempt + 1}), "
                                 f"retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All LLM call attempts failed: {str(e)}")
                    raise
    
    def retrieve_node(self, state: State) -> State:
        """
        RAG retrieval node: Get contexts for query.
        
        Args:
            state: Pipeline state
        
        Returns:
            Updated state with contexts
        """
        query = state["query"]
        
        with ContextualTimer(logger, "Retrieval stage"):
            retrieval_start = time.time()
            results = self.retriever.retrieve(query, k=RAGConfig.FINAL_RETRIEVE_K)
            retrieval_time = time.time() - retrieval_start
        
        # Extract context strings
        contexts = []
        for result in results:
            metadata = result['metadata']
            source_info = f"[{metadata.get('act_name', 'Unknown')} - "
            if metadata.get('section_number'):
                source_info += f"{metadata.get('section_number')} - "
            source_info += f"Page {metadata.get('page_number', '?')}]"
            
            contexts.append(f"{source_info}\n{result['content']}")
        
        state["contexts"] = contexts
        
        # Track metrics
        if not state.get("metadata"):
            state["metadata"] = {}
        state["metadata"]["retrieval_time"] = retrieval_time
        state["metadata"]["contexts_retrieved"] = len(contexts)
        
        logger.info(f"Retrieval complete: {len(contexts)} contexts retrieved")
        
        return state
    
    def generate_node(self, state: State) -> State:
        """
        LLM generation node: Generate legal analysis based on contexts.
        
        Args:
            state: Pipeline state
        
        Returns:
            Updated state with LLM response
        """
        query = state["query"]
        contexts = state.get("contexts", [])
        
        if not contexts:
            logger.warning("No contexts available for generation")
            state["response"] = json.dumps({
                "error": "No relevant legal information found in knowledge base",
                "note": "Insufficient context to provide legal analysis"
            })
            return state
        
        # Generate prompts
        system_prompt = self._generate_system_prompt()
        user_prompt = self._generate_user_prompt(query, contexts)
        
        try:
            # Call LLM with retry logic
            with ContextualTimer(logger, "LLM generation"):
                llm_start = time.time()
                response_text = self._call_llm_with_retry(system_prompt, user_prompt)
                llm_time = time.time() - llm_start
            
            # Validate and parse JSON
            if self.config.get('VALIDATE_JSON_OUTPUT', RAGConfig.VALIDATE_JSON_OUTPUT):
                try:
                    parsed_response = self._validate_json_response(response_text)
                    state["response"] = json.dumps(parsed_response, indent=2)
                except ResponseValidationError as e:
                    logger.error(f"JSON validation failed: {str(e)}")
                    state["response"] = json.dumps({
                        "error": "Response validation failed",
                        "raw_response": response_text[:500],
                        "note": "The LLM response could not be parsed. Please review raw response."
                    })
            else:
                state["response"] = response_text
            
            # Track metrics
            state["metadata"]["llm_time"] = llm_time
            state["metadata"]["response_length"] = len(state["response"])
            
            logger.info(f"Generation complete ({llm_time:.2f}s)")
        
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}", exc_info=True)
            state["response"] = json.dumps({
                "error": f"Generation failed: {str(e)}",
                "note": "An error occurred while generating the legal analysis"
            })
        
        return state
    
    def _initialize_pipeline(self):
        """Initialize the LanGraph pipeline."""
        graph = StateGraph(state_schema=State)
        graph.add_node("retrieve", self.retrieve_node)
        graph.add_node("generate", self.generate_node)
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        graph.set_entry_point("retrieve")
        
        self.app = graph.compile()
        logger.info("Pipeline initialized successfully")
    
    def process_query(self, query: str) -> str:
        """
        Process legal complaint through RAG pipeline.
        
        Args:
            query: Legal complaint text
        
        Returns:
            JSON string with legal analysis
        """
        if not hasattr(self, 'app'):
            self._initialize_pipeline()
        
        pipeline_start = time.time()
        
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Run pipeline
            result = self.app.invoke({
                "query": query,
                "contexts": [],
                "response": "",
                "metadata": {}
            })
            
            pipeline_time = time.time() - pipeline_start
            
            # Track metrics
            self.metrics_tracker.record_pipeline(
                total_time=pipeline_time,
                retrieval_time=result["metadata"].get("retrieval_time", 0),
                llm_time=result["metadata"].get("llm_time", 0),
                contexts_used=result["metadata"].get("contexts_retrieved", 0),
                success=True
            )
            
            # Audit log
            if self.config.get('SAVE_AUDIT_LOG', RAGConfig.SAVE_AUDIT_LOG):
                self.audit_logger.log_query_response(
                    query,
                    result["contexts"],
                    result["response"],
                    result["metadata"]
                )
            
            logger.info(f"Query processing complete ({pipeline_time:.2f}s)")
            
            return result["response"]
        
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            
            pipeline_time = time.time() - pipeline_start
            self.metrics_tracker.record_pipeline(
                total_time=pipeline_time,
                retrieval_time=0,
                llm_time=0,
                contexts_used=0,
                success=False,
                error=str(e)
            )
            
            raise


# Global LLM instance (lazy loaded)
_llm_instance = None


def get_llm_interface() -> LLMInterfaceResearch:
    """Get or create global LLM interface instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMInterfaceResearch()
    return _llm_instance


def process_query(query: str) -> str:
    """
    Process legal complaint through RAG pipeline.
    Convenience function matching original API.
    
    Args:
        query: Legal complaint text
    
    Returns:
        JSON string with legal analysis
    """
    llm = get_llm_interface()
    return llm.process_query(query)


if __name__ == "__main__":
    """
    Test LLM interface.
    """
    try:
        llm = LLMInterfaceResearch()
        
        # Test query
        test_query = "I was stolen from in a shop. What are my legal options?"
        
        print(f"Processing query: {test_query}")
        print("="*60)
        
        response = llm.process_query(test_query)
        
        print("Response:")
        print(response)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

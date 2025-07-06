import logging
from typing import List, Optional, Union
from src.llm.base import LLM

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class LlamaCppLLM(LLM):
    """LLM implementation using llama.cpp server"""
    
    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        context_window: int = 4096
    ):
        """
        Initialize the llama.cpp LLM client
        
        Args:
            server_url: URL of the llama.cpp server
            context_window: Context window size in tokens
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("Requests is not installed. Install with 'pip install requests'")
        
        self.server_url = server_url
        self.context_window = context_window
        
        # Test the connection
        try:
            response = requests.get(f"{server_url}/health")
            if response.status_code == 200:
                logger.info(f"Connected to llama.cpp server at {server_url}")
            else:
                logger.warning(f"llama.cpp server at {server_url} returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to connect to llama.cpp server: {e}")
    
    def generate(
        self, 
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate text from a prompt using llama.cpp server"""
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": stop_sequences if stop_sequences else []
        }
        
        response = requests.post(
            f"{self.server_url}/completion",
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"LLM request failed with status {response.status_code}: {response.text}")
        
        result = response.json()
        
        # Extract the generated text
        return result.get("content", "").strip()
    
    def generate_with_context(
        self,
        question: str,
        context: Union[str, List[str]],
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """Generate an answer based on a question and context"""
        # Format the prompt with context
        prompt = self.format_prompt_with_context(question, context)
        
        # Generate the response
        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    def format_prompt_with_context(
        self,
        question: str,
        context: Union[str, List[str]]
    ) -> str:
        """Format a RAG prompt with context for llama.cpp"""
        # Convert context list to string if needed
        if isinstance(context, list):
            context_str = "\n\n".join(context)
        else:
            context_str = context
        
        # Build a prompt for llama models
        prompt = (
            "You are an intelligent assistant that provides accurate information "
            "based on the provided context. Answer the question using only the information "
            "from the context. If you cannot find the answer in the context, say 'I don't have "
            "enough information to answer this question.' Don't make up information that "
            "isn't directly supported by the context.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer: "
        )
        
        return prompt
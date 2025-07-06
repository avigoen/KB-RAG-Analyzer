import logging
from typing import List, Optional, Union
from src.llm.base import LLM

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class OpenAICompatibleLLM(LLM):
    """LLM implementation using an OpenAI-compatible API"""
    
    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        model_name: str = "llama2"
    ):
        """
        Initialize an OpenAI-compatible API client
        
        Args:
            api_url: URL of the API endpoint
            api_key: Optional API key
            model_name: Name of the model to use
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("Requests is not installed. Install with 'pip install requests'")
        
        self.api_url = api_url
        self.model_name = model_name
        
        # Headers for API requests
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        logger.info(f"Initialized OpenAICompatibleLLM with endpoint '{api_url}'")
    
    def generate(
        self, 
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate text from a prompt"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        response = requests.post(
            f"{self.api_url}/completions",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"LLM request failed with status {response.status_code}: {response.text}")
        
        result = response.json()
        
        # Extract the generated text
        return result.get("choices", [{}])[0].get("text", "").strip()
    
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
        """Format a RAG prompt with context"""
        # Convert context list to string if needed
        if isinstance(context, list):
            context_str = "\n\n".join(context)
        else:
            context_str = context
        
        # Build a generic prompt
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
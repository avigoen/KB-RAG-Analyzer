from typing import Optional

from .base import LLM


class LLMFactory:
    """Factory for creating LLM instances"""
    
    @staticmethod
    def create_llm(
        llm_type: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> LLM:
        """
        Create an LLM of the specified type
        
        Args:
            llm_type: Type of LLM ("huggingface", "llamacpp", "openai-compatible")
            model_name: Name of the model to use (depends on llm_type)
            **kwargs: Additional arguments for the specific LLM type
            
        Returns:
            LLM instance
        """
        if llm_type == "huggingface":
            from .transformers import HuggingFaceTransformersLLM
            return HuggingFaceTransformersLLM(
                model_name=model_name or "mistralai/Mistral-7B-Instruct-v0.2",
                **kwargs
            )
        elif llm_type == "llamacpp":
            from .llamacpp import LlamaCppLLM
            return LlamaCppLLM(
                server_url=kwargs.get("server_url", "http://localhost:8080"),
                **{k: v for k, v in kwargs.items() if k != "server_url"}
            )
        elif llm_type == "openai-compatible":
            from .openai import OpenAICompatibleLLM
            return OpenAICompatibleLLM(
                api_url=kwargs.get("api_url", "http://localhost:8000/v1"),
                api_key=kwargs.get("api_key"),
                model_name=model_name or "llama2",
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}") 
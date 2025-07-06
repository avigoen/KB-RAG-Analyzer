from abc import ABC, abstractmethod
from typing import List, Optional, Union


class LLM(ABC):
    """Abstract base class for LLM interfaces"""
    
    @abstractmethod
    def generate(
        self, 
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate text from a prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            stop_sequences: Optional list of sequences that stop generation
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def generate_with_context(
        self,
        question: str,
        context: Union[str, List[str]],
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Generate an answer based on a question and context
        
        Args:
            question: Question to answer
            context: Context information (string or list of strings)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated answer
        """
        pass
    
    @abstractmethod
    def format_prompt_with_context(
        self,
        question: str,
        context: Union[str, List[str]]
    ) -> str:
        """
        Format a RAG prompt with context
        
        Args:
            question: Question to answer
            context: Context information (string or list of strings)
            
        Returns:
            Formatted prompt string
        """
        pass
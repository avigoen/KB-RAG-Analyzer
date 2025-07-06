import logging
from typing import List, Optional, Union
from src.llm.base import LLM

try:
    import torch
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        pipeline,
        TextIteratorStreamer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class HuggingFaceTransformersLLM(LLM):
    """LLM implementation using HuggingFace Transformers"""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device_map: str = "auto",
        load_in_8bit: bool = True,
        max_context_length: int = 4096,
        trust_remote_code: bool = False
    ):
        """
        Initialize the HuggingFace Transformers LLM
        
        Args:
            model_name: Name of the HF model to use
            device_map: Device mapping strategy
            load_in_8bit: Whether to quantize model to 8-bit
            max_context_length: Maximum context length in tokens
            trust_remote_code: Whether to trust remote code
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers is not installed. Install with 'pip install transformers accelerate torch'")
        
        self.model_name = model_name
        self.max_context_length = max_context_length
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        
        # Configure model loading options
        model_kwargs = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Create text generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        logger.info(f"Initialized HuggingFaceTransformersLLM with model '{model_name}'")
    
    def generate(
        self, 
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate text from a prompt"""
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id
        }
        
        # Add stop sequences if provided
        if stop_sequences:
            generation_kwargs["stopping_criteria"] = self._create_stopping_criteria(prompt, stop_sequences)
        
        outputs = self.pipeline(
            prompt,
            **generation_kwargs
        )
        
        # Extract the generated text (excluding the prompt)
        generated_text = outputs[0]["generated_text"][len(prompt):]
        
        # Remove any stop sequences from the end of the text
        if stop_sequences:
            for stop_seq in stop_sequences:
                if generated_text.endswith(stop_seq):
                    generated_text = generated_text[:-len(stop_seq)]
        
        return generated_text.strip()
    
    def _create_stopping_criteria(self, prompt: str, stop_sequences: List[str]):
        """Create stopping criteria for generation"""
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        class StopSequenceCriteria(StoppingCriteria):
            def __init__(self, tokenizer, stop_sequences, prompt_length):
                self.tokenizer = tokenizer
                self.stop_sequences = stop_sequences
                self.prompt_length = prompt_length
                
            def __call__(self, input_ids, scores, **kwargs):
                # Get the generated text so far
                generated_text = self.tokenizer.decode(input_ids[0][self.prompt_length:])
                
                # Check if any stop sequence appears at the end of the generated text
                for stop_seq in self.stop_sequences:
                    if generated_text.endswith(stop_seq):
                        return True
                
                return False
        
        # Get the token IDs for the prompt
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)
        
        criteria = StopSequenceCriteria(self.tokenizer, stop_sequences, prompt_length)
        return StoppingCriteriaList([criteria])
    
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
        
        # Build a prompt that works well with instruction-following models
        prompt = (
            f"<s>[INST] You are an intelligent assistant that provides accurate information "
            f"based on the provided context. Answer the question using only the information "
            f"from the context. If you cannot find the answer in the context, say 'I don't have "
            f"enough information to answer this question.' Don't make up information that "
            f"isn't directly supported by the context.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question} [/INST]"
        )
        
        return prompt

"""
Inference pipeline interfaces for the HADES modular pipeline architecture.

This module defines the interfaces for the inference pipeline components,
including prompt templates, context assembly, and response formatting.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Generator

from hades.pipeline.interfaces import Component, Configurable, Pipeline, Pluggable, Serializable
from hades.retrieve.interface import Query, RetrievalResult, RetrievalPath


class Prompt:
    """Class representing a prompt in the inference pipeline."""
    
    def __init__(self, prompt_id: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Initialize a prompt.
        
        Args:
            prompt_id: Unique identifier for the prompt
            text: Prompt text
            metadata: Optional metadata for the prompt
        """
        self.prompt_id = prompt_id
        self.text = text
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"Prompt(id={self.prompt_id}, text='{self.text[:30]}...')"


class Context:
    """Class representing a context in the inference pipeline."""
    
    def __init__(self, context_id: str, text: str, 
                 source_results: Union[List[RetrievalResult], List[RetrievalPath]], 
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize a context.
        
        Args:
            context_id: Unique identifier for the context
            text: Context text
            source_results: Source retrieval results or paths
            metadata: Optional metadata for the context
        """
        self.context_id = context_id
        self.text = text
        self.source_results = source_results
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"Context(id={self.context_id}, length={len(self.text)})"


class Response:
    """Class representing a response in the inference pipeline."""
    
    def __init__(self, response_id: str, text: str, 
                 prompt: Prompt, context: Optional[Context] = None, 
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize a response.
        
        Args:
            response_id: Unique identifier for the response
            text: Response text
            prompt: Prompt that generated this response
            context: Optional context used for generation
            metadata: Optional metadata for the response
        """
        self.response_id = response_id
        self.text = text
        self.prompt = prompt
        self.context = context
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"Response(id={self.response_id}, length={len(self.text)})"


class PromptTemplate(Component, Configurable, Pluggable, ABC):
    """Interface for prompt templates in the inference pipeline."""
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "prompt_template"
    
    @classmethod
    def get_plugin_type(cls) -> str:
        """Return the plugin type for this component."""
        return "prompt_template"
    
    @abstractmethod
    def format_prompt(self, query: Query, context: Optional[Context] = None, 
                     metadata: Optional[Dict[str, Any]] = None) -> Prompt:
        """Format a prompt from a query and optional context.
        
        Args:
            query: Query to format prompt for
            context: Optional context to include in the prompt
            metadata: Optional metadata for the prompt
            
        Returns:
            Formatted prompt
        """
        pass
    
    @abstractmethod
    def get_template(self) -> str:
        """Return the template string.
        
        Returns:
            Template string
        """
        pass
    
    @abstractmethod
    def set_template(self, template: str) -> None:
        """Set the template string.
        
        Args:
            template: Template string
        """
        pass


class ContextAssembler(Component, Configurable, Pluggable, ABC):
    """Interface for context assemblers in the inference pipeline."""
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "context_assembler"
    
    @classmethod
    def get_plugin_type(cls) -> str:
        """Return the plugin type for this component."""
        return "context_assembler"
    
    @abstractmethod
    def assemble_context(self, query: Query, 
                        retrieval_results: Union[List[RetrievalResult], List[RetrievalPath]], 
                        max_length: Optional[int] = None, 
                        metadata: Optional[Dict[str, Any]] = None) -> Context:
        """Assemble a context from retrieval results.
        
        Args:
            query: Query to assemble context for
            retrieval_results: Retrieval results or paths to assemble context from
            max_length: Optional maximum length for the context
            metadata: Optional metadata for the context
            
        Returns:
            Assembled context
        """
        pass


class ResponseFormatter(Component, Configurable, Pluggable, ABC):
    """Interface for response formatters in the inference pipeline."""
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "response_formatter"
    
    @classmethod
    def get_plugin_type(cls) -> str:
        """Return the plugin type for this component."""
        return "response_formatter"
    
    @abstractmethod
    def format_response(self, response: Response) -> str:
        """Format a response.
        
        Args:
            response: Response to format
            
        Returns:
            Formatted response string
        """
        pass
    
    @abstractmethod
    def format_streaming_response(self, response_stream: Generator[str, None, None], 
                                 prompt: Prompt, 
                                 context: Optional[Context] = None) -> Generator[str, None, None]:
        """Format a streaming response.
        
        Args:
            response_stream: Stream of response chunks
            prompt: Prompt that generated the response
            context: Optional context used for generation
            
        Returns:
            Stream of formatted response chunks
        """
        pass


class InferencePipeline(Pipeline, ABC):
    """Interface for the inference pipeline."""
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "inference_pipeline"
    
    @abstractmethod
    def generate(self, query_text: str, 
                retrieval_results: Optional[Union[List[RetrievalResult], List[RetrievalPath]]] = None, 
                stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate a response for a query with optional retrieval results.
        
        Args:
            query_text: Query text
            retrieval_results: Optional retrieval results or paths
            stream: Whether to stream the response
            
        Returns:
            Generated response or stream of response chunks
        """
        pass
    
    @abstractmethod
    def retrieve_and_generate(self, query_text: str, 
                            top_k: int = 10, 
                            use_paths: bool = True, 
                            stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Retrieve and generate a response for a query.
        
        Args:
            query_text: Query text
            top_k: Number of retrieval results to use
            use_paths: Whether to use path finding
            stream: Whether to stream the response
            
        Returns:
            Generated response or stream of response chunks
        """
        pass
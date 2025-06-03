"""
Inference request and response types for HADES-PathRAG model engines.

This module provides centralized type definitions for inference requests and responses
used by model engines throughout the HADES-PathRAG system.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union, Literal


class CompletionRequest(TypedDict, total=False):
    """Request for text completion."""
    model: str
    prompt: Union[str, List[str]]
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    stop: Optional[Union[str, List[str]]]
    echo: bool
    stream: bool
    logprobs: Optional[int]
    presence_penalty: float
    frequency_penalty: float
    repetition_penalty: float
    best_of: int
    n: int
    logit_bias: Dict[int, float]


class ChatMessage(TypedDict):
    """Message in a chat completion request."""
    role: Literal["system", "user", "assistant", "function"]
    content: str
    name: Optional[str]


class ChatCompletionRequest(TypedDict, total=False):
    """Request for chat completion."""
    model: str
    messages: List[ChatMessage]
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    stop: Optional[Union[str, List[str]]]
    stream: bool
    presence_penalty: float
    frequency_penalty: float
    repetition_penalty: float
    n: int
    logit_bias: Dict[int, float]
    user: str
    response_format: Dict[str, str]


class EmbeddingRequest(TypedDict, total=False):
    """Request for text embedding."""
    model: str
    input: Union[str, List[str]]
    encoding_format: str
    user: str
    dimensions: int
    normalize: bool


class CompletionChoice(TypedDict, total=False):
    """A single completion choice."""
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]]
    finish_reason: str


class CompletionResponse(TypedDict, total=False):
    """Response from a completion request."""
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]


class ChatCompletionChoice(TypedDict, total=False):
    """A single chat completion choice."""
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(TypedDict, total=False):
    """Response from a chat completion request."""
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


class EmbeddingData(TypedDict, total=False):
    """Embedding data for a single input."""
    object: str
    embedding: List[float]
    index: int


class EmbeddingResponse(TypedDict, total=False):
    """Response from an embedding request."""
    object: str
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, int]


# Common type for all inference requests
InferenceRequest = Union[CompletionRequest, ChatCompletionRequest, EmbeddingRequest]

# Common type for all inference responses
InferenceResponse = Union[CompletionResponse, ChatCompletionResponse, EmbeddingResponse]

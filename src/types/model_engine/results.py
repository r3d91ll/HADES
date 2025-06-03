"""
Result types for model engine adapters in HADES-PathRAG.

This module defines the standardized result types returned by model engine adapters,
ensuring consistent structure and type safety across different adapter implementations.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union, Literal
from enum import Enum


class ChatRole(str, Enum):
    """Roles in a chat conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class EmbeddingResult(TypedDict, total=False):
    """Result from an embedding operation."""
    embeddings: List[List[float]]
    metadata: Dict[str, Any]


class CompletionMetadata(TypedDict, total=False):
    """Metadata for a completion result."""
    model_id: str
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float


class CompletionResult(TypedDict, total=False):
    """Result from a text completion operation."""
    text: str
    metadata: CompletionMetadata


class ChatMessage(TypedDict, total=False):
    """Message in a chat conversation."""
    role: str
    content: str
    name: Optional[str]


class ChatCompletionMetadata(TypedDict, total=False):
    """Metadata for a chat completion result."""
    model_id: str
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    finish_reason: Optional[str]


class ChatCompletionResult(TypedDict, total=False):
    """Result from a chat completion operation."""
    message: ChatMessage
    metadata: ChatCompletionMetadata


class StreamingChatCompletionResult(TypedDict, total=False):
    """Result chunk from a streaming chat completion operation."""
    chunk: str
    finish_reason: Optional[str]
    metadata: ChatCompletionMetadata


# Option types for adapter operations
class EmbeddingOptions(TypedDict, total=False):
    """Options for embedding operations."""
    dimensions: Optional[int]
    normalize: bool
    encoding_format: str


class CompletionOptions(TypedDict, total=False):
    """Options for completion operations."""
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    stop: Optional[Union[str, List[str]]]
    echo: bool
    stream: bool
    presence_penalty: float
    frequency_penalty: float
    repetition_penalty: float
    best_of: int
    logit_bias: Optional[Dict[int, float]]


class ChatCompletionOptions(TypedDict, total=False):
    """Options for chat completion operations."""
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    stop: Optional[Union[str, List[str]]]
    stream: bool
    presence_penalty: float
    frequency_penalty: float
    repetition_penalty: float
    logit_bias: Optional[Dict[int, float]]
    response_format: Optional[Dict[str, str]]

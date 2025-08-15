"""
Data model definition module

Uses Pydantic to define OpenAI format request and response models
"""

from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field
import time
import uuid


class ChatContentPart(BaseModel):
    """Chat content part model for vision support"""
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None  # {"url": "..."}


class ChatMessage(BaseModel):
    """Chat message model"""
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[ChatContentPart]]


class ChatCompletionRequest(BaseModel):
    """Chat completion request model"""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    stream: Optional[bool] = False
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    """Chat completion choice model"""
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter"] = "stop"


class ChatCompletionResponse(BaseModel):
    """Chat completion response model"""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]


class ChatCompletionStreamChoice(BaseModel):
    """Streaming chat completion choice model"""
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None


class ChatCompletionStreamResponse(BaseModel):
    """Streaming chat completion response model"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class ErrorDetail(BaseModel):
    """Error detail model"""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model"""
    error: ErrorDetail


class ModelInfo(BaseModel):
    """Model information model"""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "qwen-code-proxy"


class ModelsResponse(BaseModel):
    """Models list response model"""
    object: str = "list"
    data: List[ModelInfo]


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = "ok"
    version: str
    timestamp: int = Field(default_factory=lambda: int(time.time())) 
"""Pydantic models for API"""
from pydantic import BaseModel, Field
from typing import Optional

class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(512, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    use_rag: bool = Field(False, description="Whether to use RAG for context retrieval")

class GenerationResponse(BaseModel):
    text: str = Field(..., description="Generated text")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    latency_ms: float = Field(..., description="Generation latency in milliseconds")
    context_used: Optional[bool] = Field(None, description="Whether RAG context was used")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    device: str = Field(..., description="Device being used (cuda/cpu)")


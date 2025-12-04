"""Tests for inference module"""
import pytest
from src.api.inference import ModelServer
import os

@pytest.mark.skipif(
    not os.path.exists("./models/finetuned/medical-llm"),
    reason="Model not available for testing"
)
def test_model_loading():
    """Test model loading"""
    model_server = ModelServer(model_path="./models/finetuned/medical-llm")
    assert model_server.model is not None
    assert model_server.tokenizer is not None

@pytest.mark.skipif(
    not os.path.exists("./models/finetuned/medical-llm"),
    reason="Model not available for testing"
)
def test_generation():
    """Test text generation"""
    model_server = ModelServer(model_path="./models/finetuned/medical-llm")
    result = model_server.generate(
        prompt="What is AI?",
        max_tokens=50,
        temperature=0.7,
        top_p=0.9
    )
    assert "text" in result
    assert "tokens_generated" in result
    assert "latency_ms" in result
    assert result["tokens_generated"] > 0


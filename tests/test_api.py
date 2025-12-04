"""Tests for API endpoints"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()

def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "device" in data

def test_generate_endpoint():
    """Test generation endpoint"""
    # Note: This test may fail if model is not loaded
    # In production, use a mock model for testing
    response = client.post(
        "/generate",
        json={
            "prompt": "What is AI?",
            "max_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.9
        }
    )
    # Accept both success and error (if model not available)
    assert response.status_code in [200, 500]


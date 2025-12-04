"""API module for serving the fine-tuned model"""
from .main import app
from .inference import ModelServer

__all__ = ["app", "ModelServer"]


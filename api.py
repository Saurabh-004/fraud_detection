"""Backward-compatible entry point. Prefer: uvicorn src.api:app"""
from src.api import app

__all__ = ["app"]

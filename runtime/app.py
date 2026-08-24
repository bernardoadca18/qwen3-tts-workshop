"""Compatibility import for historical uvicorn app:app commands.

Model and CUDA initialization occur only in the FastAPI lifespan.
"""

from voice_service.main import app

__all__ = ["app"]

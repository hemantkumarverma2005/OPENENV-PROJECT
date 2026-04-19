"""
server/app.py
─────────────
OpenEnv multi-mode deployment entry point.
Re-exports the FastAPI app from the root-level app module.
"""

import sys
import os

# Ensure root project directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the FastAPI app from root app.py
from app import app  # noqa: F401

import uvicorn


def main():
    """Entry point for `server` script defined in pyproject.toml."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()

"""
AI Search Engine API - Main Entry Point

This is the main entry point for the AI Search Engine API.
It imports the FastAPI application from the api module and provides
a clean interface for deployment systems.

Author: AI Search Engine Team
Version: 1.0.0
"""

from api.index import app

# Export the FastAPI app for deployment systems
__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True,  # Enable auto-reload in development
        log_level="info"
    )

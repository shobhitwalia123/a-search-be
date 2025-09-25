"""
AI Search Engine API - Vercel Handler

This module provides the Vercel serverless function handler for the AI Search Engine API.
It imports the FastAPI application and exports it as 'handler' for Vercel deployment.

Author: AI Search Engine Team
Version: 1.0.0
"""

from index import app

# Export the FastAPI app as 'handler' for Vercel
handler = app

# Main FastAPI entrypoint for deployment
from api.index import app

# Export the app for deployment systems
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

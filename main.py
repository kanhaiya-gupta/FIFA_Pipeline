from fastapi import FastAPI
from src.api.app import create_app

# Initialize FastAPI app
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.moderation import router as moderation_router
import os

app = FastAPI(
    title="Aegis AI Backend",
    description="Multimodal video moderation API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    moderation_router,
    prefix="/api"
)

@app.get("/")
def root():
    return {
        "message": "Aegis AI backend is running 🚀"
    }

os.makedirs("uploads", exist_ok=True)
os.makedirs("temp", exist_ok=True)
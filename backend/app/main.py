# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat
import os
from pathlib import Path
from dotenv import load_dotenv
current_dir = Path(__file__).resolve().parent
env_path = current_dir.parent / '.env'
load_dotenv(dotenv_path=env_path)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="Legal AI Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")

@app.get("/")
async def root():
    return {"status": "Legal AI API is running 🚀"}
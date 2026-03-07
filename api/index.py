import os
import sys
import glob
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from google import genai
from pypdf import PdfReader
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY")

# Set base path for data folder relative to this file
# In Vercel, the directory structure is preserved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# Models
EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODEL = "models/gemini-3-flash-preview" # Use a stable known model

# Backend Client
client = genai.Client(api_key=API_KEY) if API_KEY else None

# Global memory for RAG
knowledge_base = []

def load_knowledge_base():
    """Reads all PDFs in the data directory and generates embeddings."""
    global knowledge_base
    if knowledge_base:
        return
    knowledge_base = []
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        return

    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_files:
        print("No PDF files found in data directory.")
        return

    print(f"Processing {len(pdf_files)} PDF(s)...")
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            filename = os.path.basename(pdf_path)
            
            full_text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
            
            paragraphs = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 50]
            
            for chunk in paragraphs:
                try:
                    if client:
                        embed_res = client.models.embed_content(
                            model=EMBEDDING_MODEL,
                            contents=chunk
                        )
                        embedding = embed_res.embeddings[0].values
                        
                        knowledge_base.append({
                            "content": chunk,
                            "embedding": embedding,
                            "source": filename
                        })
                except Exception as e:
                    print(f"Error embedding chunk from {filename}: {e}")
                    
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")

    print(f"Successfully loaded {len(knowledge_base)} chunks into memory.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_knowledge_base()
    yield
    knowledge_base.clear()

app = FastAPI(title="AgroMart RAG Service", lifespan=lifespan)

# CORS - Allow all for flexibility in dev/prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

def get_cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_context(query, top_k=4):
    if not knowledge_base or not client:
        return ""
    
    try:
        query_res = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query
        )
        query_embedding = query_res.embeddings[0].values
        
        scores = []
        for item in knowledge_base:
            similarity = get_cosine_similarity(query_embedding, item["embedding"])
            scores.append((similarity, item))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [f"[Source: {s[1]['source']}] {s[1]['content']}" for s in scores[:top_k]]
        return "\n\n".join(top_chunks)
    except Exception as e:
        print(f"Search error: {e}")
        return ""

@app.post("/chat")
async def chat(request: ChatRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Gemini Client not initialized. Check API Key.")
    
    try:
        # For Vercel, ensure knowledge base is loaded if not already 
        # (Lifespan might behave differently in some serverless modes)
        if not knowledge_base:
            load_knowledge_base()
            
        context = find_context(request.message)
        
        system_prompt = f"""
You are **AgroMart AI**, an expert agricultural assistant.

RESPONSE PROTOCOL:
1. Greetings: Respond warmly.
2. Context-Based: Use ONLY provided context for technical info.
3. Sources: Use "According to our official guides...".
4. No Context: Say you don't have that info in the guides.
5. Formatting: Use bold text for key points. No ***.

DATABASE CONTEXT:
{context if context else "No relevant data found."}
"""

        def stream_gen():
            try:
                response_stream = client.models.generate_content_stream(
                    model=GENERATION_MODEL,
                    contents=f"{system_prompt}\n\nUser Question: {request.message}"
                )
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
            except Exception as e:
                yield f"AI Service Error: {str(e)}"

        return StreamingResponse(stream_gen(), media_type="text/plain")
        
    except Exception as e:
        def err_gen(): yield f"Service Error: {str(e)}"
        return StreamingResponse(err_gen(), media_type="text/plain")

@app.get("/api/health")
async def root():
    return {"status": "running", "chunks": len(knowledge_base)}

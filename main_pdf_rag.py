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
if not API_KEY and not os.getenv("VERCEL"):
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Set base path for data folder relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Verified compatible models
EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODEL = "models/gemini-3-flash-preview"

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
            
            # Extract text from all pages
            full_text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
            
            # Simple chunking logic: split by double newline or fixed length
            # Aim for chunks around 500-1000 characters
            paragraphs = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 50]
            
            for chunk in paragraphs:
                try:
                    # Generate embedding
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
    # Startup: Load PDFs
    load_knowledge_base()
    yield
    # Shutdown: Clear memory
    knowledge_base.clear()

app = FastAPI(title="AgroMart RAG Service", lifespan=lifespan)

# CORS
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
    """Finds most relevant chunks based on cosine similarity."""
    if not knowledge_base:
        return ""
    
    try:
        # Embed query
        query_res = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query
        )
        query_embedding = query_res.embeddings[0].values
        
        # Calculate similarities
        scores = []
        for item in knowledge_base:
            similarity = get_cosine_similarity(query_embedding, item["embedding"])
            scores.append((similarity, item))
        
        # Sort and take top_k
        scores.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [f"[Source: {s[1]['source']}] {s[1]['content']}" for s in scores[:top_k]]
        return "\n\n".join(top_chunks)
    except Exception as e:
        print(f"Search error: {e}")
        return ""

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        context = find_context(request.message)
        
        system_prompt = f"""
You are **AgroMart AI**, an expert agricultural assistant.

========================
RESPONSE PROTOCOL
========================
1.  **Greetings**: If the user says hello/hi, respond warmly as AgroMart AI.
2.  **Context-Based Answers**: For all technical questions, use ONLY the provided context.
3.  **Source Formatting**: Start your factual response with "According to our official guides...".
4.  **No Context**: If the answer isn't in the guides, say: "I'm sorry, I don't have information on that in our current guides."
5.  **Conciseness**: Keep it brief and well-formatted with markdown.

========================
DATABASE CONTEXT:
========================
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
                print(f"Streaming error: {e}")
                yield f"AI Service Error: {str(e)}"

        return StreamingResponse(stream_gen(), media_type="text/plain")
        
    except Exception as e:
        print(f"Chat error: {e}")
        def err_gen(): yield f"Service Error: {str(e)}"
        return StreamingResponse(err_gen(), media_type="text/plain")

@app.get("/")
async def root():
    return {"status": "running", "chunks_loaded": len(knowledge_base)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

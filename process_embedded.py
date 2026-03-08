import os
import json
import glob
from google import genai
from pypdf import PdfReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "knowledge_base.json")

# Verified compatible models
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Backend Client
client = genai.Client(api_key=API_KEY)

def generate_knowledge_base():
    """Reads all PDFs in the data directory and generates embeddings, saving to JSON."""
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        return

    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_files:
        print("No PDF files found in data directory.")
        return

    knowledge_base = []
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
            
            # Simple chunking logic: split by double newline or fixed length
            chunks = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 50]
            
            print(f"Generating embeddings for {filename} ({len(chunks)} chunks)...")
            for chunk in chunks:
                try:
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
                    print(f"  Error embedding chunk from {filename}: {e}")
                    
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")

    # Save to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)

    print(f"Successfully generated {len(knowledge_base)} chunks and saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    generate_knowledge_base()

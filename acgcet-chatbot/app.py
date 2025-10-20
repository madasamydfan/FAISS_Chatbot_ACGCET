import json
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import random

app = FastAPI(title="ACGCET Chatbot", version="2.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load intents file
with open("intents.json", "r", encoding="utf-8") as f:
    intents_data = json.load(f)

# Prepare corpus, responses, tags, and links
corpus, responses, tags, links = [], [], [], []

for intent in intents_data["intents"]:
    for pattern in intent["patterns"]:
        corpus.append(pattern)
        responses.append(random.choice(intent["responses"]))
        tags.append(intent["tag"])

        if "href=" in intent["responses"][0]:
            start = intent["responses"][0].find("href=") + 6
            end = intent["responses"][0].find('"', start)
            links.append(intent["responses"][0][start:end])
        else:
            links.append(None)

# Globals
model = None
index = None
EMBEDDING_FILE = "embeddings.npy"
INDEX_FILE = "faiss_index.bin"


def generate_embeddings_and_index(force: bool = False):
    """
    Generate embeddings and FAISS index from corpus.
    If force=True, regenerate even if files exist.
    """
    global model, index

    if model is None:
        print("🔹 Loading model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

    # Check if files exist
    embeddings_exist = os.path.exists(EMBEDDING_FILE)
    index_exist = os.path.exists(INDEX_FILE)

    if embeddings_exist and index_exist and not force:
        print("✅ Embeddings and FAISS index already exist. Skipping generation.")
        return

    print("⚙️ Generating embeddings and FAISS index...")
    embeddings = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
    np.save(EMBEDDING_FILE, embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    print("✅ Embeddings and FAISS index saved!")


def load_index():
    """Load FAISS index from file."""
    global index
    if index is None:
        if not os.path.exists(INDEX_FILE) or not os.path.exists(EMBEDDING_FILE):
            print("⚠️ Embeddings or index not found. Generating now...")
            generate_embeddings_and_index()
        embeddings = np.load(EMBEDDING_FILE)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index = faiss.read_index(INDEX_FILE)
        print("✅ FAISS index loaded.")


@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    query = data.get("query", "").strip()

    if not query:
        return {"error": "Query is required."}

    # Ensure model and index are loaded
    if model is None:
        generate_embeddings_and_index()
    if index is None:
        load_index()

    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(query_emb, k=1)
    best_idx = I[0][0]

    if D[0][0] < 0.3:  # similarity threshold
        return {
            "query": query,
            "intent": "fallback",
            "response": "I'm not sure about that. Could you rephrase or ask something about ACGCET?",
            "similarity": float(D[0][0])
        }

    return {
        "query": query,
        "intent": tags[best_idx],
        "response": responses[best_idx],
        "similarity": float(D[0][0]),
        "navigation_link": links[best_idx],
    }


@app.get("/generate_embeddings")
def generate_embeddings_endpoint(force: bool = False):
    """Optional endpoint to regenerate embeddings/index manually."""
    generate_embeddings_and_index(force)
    return {"status": "Embeddings and index generated successfully!"}


@app.get("/")
def home():
    return {"message": "ACGCET Chatbot is running 🚀"}

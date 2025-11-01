# import json
# import os
# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import random

# app = FastAPI(title="ACGCET Chatbot", version="2.0")

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load intents file
# with open("intents.json", "r", encoding="utf-8") as f:
#     intents_data = json.load(f)

# # Prepare corpus, responses, tags, and links
# corpus, responses, tags, links = [], [], [], []

# for intent in intents_data["intents"]:
#     for pattern in intent["patterns"]:
#         corpus.append(pattern)
#         responses.append(random.choice(intent["responses"]))
#         tags.append(intent["tag"])

#         if "href=" in intent["responses"][0]:
#             start = intent["responses"][0].find("href=") + 6
#             end = intent["responses"][0].find('"', start)
#             links.append(intent["responses"][0][start:end])
#         else:
#             links.append(None)

# # Globals
# model = None
# index = None
# EMBEDDING_FILE = "embeddings.npy"
# INDEX_FILE = "faiss_index.bin"


# def generate_embeddings_and_index(force: bool = False):
#     """
#     Generate embeddings and FAISS index from corpus.
#     If force=True, regenerate even if files exist.
#     """
#     global model, index

#     if model is None:
#         print("🔹 Loading model...")
#         model = SentenceTransformer("all-MiniLM-L6-v2")

#     # Check if files exist
#     embeddings_exist = os.path.exists(EMBEDDING_FILE)
#     index_exist = os.path.exists(INDEX_FILE)

#     if embeddings_exist and index_exist and not force:
#         print("✅ Embeddings and FAISS index already exist. Skipping generation.")
#         return

#     print("⚙️ Generating embeddings and FAISS index...")
#     embeddings = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
#     np.save(EMBEDDING_FILE, embeddings)

#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     index.add(embeddings)
#     faiss.write_index(index, INDEX_FILE)
#     print("✅ Embeddings and FAISS index saved!")


# def load_index():
#     """Load FAISS index from file."""
#     global index
#     if index is None:
#         if not os.path.exists(INDEX_FILE) or not os.path.exists(EMBEDDING_FILE):
#             print("⚠️ Embeddings or index not found. Generating now...")
#             generate_embeddings_and_index()
#         embeddings = np.load(EMBEDDING_FILE)
#         dim = embeddings.shape[1]
#         index = faiss.IndexFlatIP(dim)
#         index = faiss.read_index(INDEX_FILE)
#         print("✅ FAISS index loaded.")


# @app.post("/ask")
# async def ask(request: Request):
#     data = await request.json()
#     query = data.get("query", "").strip()

#     if not query:
#         return {"error": "Query is required."}

#     # Ensure model and index are loaded
#     if model is None:
#         generate_embeddings_and_index()
#     if index is None:
#         load_index()

#     query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
#     D, I = index.search(query_emb, k=1)
#     best_idx = I[0][0]

#     if D[0][0] < 0.3:  # similarity threshold
#         return {
#             "query": query,
#             "intent": "fallback",
#             "response": "I'm not sure about that. Could you rephrase or ask something about ACGCET?",
#             "similarity": float(D[0][0])
#         }

#     return {
#         "query": query,
#         "intent": tags[best_idx],
#         "response": responses[best_idx],
#         "similarity": float(D[0][0]),
#         "navigation_link": links[best_idx],
#     }


# @app.get("/generate_embeddings")
# def generate_embeddings_endpoint(force: bool = True):
#     """Optional endpoint to regenerate embeddings/index manually."""
#     generate_embeddings_and_index(force)
#     return {"status": "Embeddings and index generated successfully!"}


# @app.get("/")
# def home():
#     return {"message": "ACGCET Chatbot is running 🚀"}

# #  uvicorn app:app --reload --host 0.0.0.0 --port 8000 
import json
import os
import re
import logging
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import random
from threading import Lock

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("acgcet-chatbot")

app = FastAPI(title="ACGCET Chatbot", version="2.0")

# CORS — in production restrict allow_origins to your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: replace with ["https://your.frontend.domain"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Files
INTENTS_FILE = os.environ.get("INTENTS_FILE", "intents.json")
EMBEDDING_FILE = os.environ.get("EMBEDDING_FILE", "embeddings.npy")
INDEX_FILE = os.environ.get("INDEX_FILE", "faiss_index.bin")

# Globals (singletons)
model = None
index = None
corpus = []
tags = []
# We'll keep a mapping from tag -> list of responses and tag -> link (if exists)
tag_to_responses = {}
tag_to_link = {}
emb_lock = Lock()

# Similarity threshold (default 0.5)
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", 0.5))


def load_intents():
    """Load intents.json into memory and prepare corpus and tag mappings."""
    global corpus, tags, tag_to_responses, tag_to_link
    logger.info("Loading intents from %s", INTENTS_FILE)
    with open(INTENTS_FILE, "r", encoding="utf-8") as f:
        intents_data = json.load(f)

    corpus = []
    tags = []
    tag_to_responses = {}
    tag_to_link = {}

    for intent in intents_data.get("intents", []):
        tag = intent.get("tag")
        patterns = intent.get("patterns", [])
        responses = intent.get("responses", [])
        # store patterns into corpus
        for patt in patterns:
            corpus.append(patt)
            tags.append(tag)
        # store responses mapping
        tag_to_responses[tag] = responses if isinstance(responses, list) else [str(responses)]

        # Extract first href if any using regex (safer)
        link = None
        if tag_to_responses[tag]:
            m = re.search(r'href=["\']([^"\']+)["\']', tag_to_responses[tag][0])
            if m:
                link = m.group(1)
        tag_to_link[tag] = link

    logger.info("Loaded %d corpus patterns and %d tags", len(corpus), len(tag_to_responses))


def generate_embeddings_and_index(force: bool = False):
    """
    Generate embeddings and FAISS index from corpus.
    If force=True, regenerate even if files exist.
    """
    global model, index
    with emb_lock:
        if model is None:
            logger.info("Loading SentenceTransformer model...")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Model loaded.")

        embeddings_exist = os.path.exists(EMBEDDING_FILE)
        index_exist = os.path.exists(INDEX_FILE)

        if embeddings_exist and index_exist and not force:
            logger.info("Embeddings and index exist and force=False -> skipping generation")
            return

        if not corpus:
            raise RuntimeError("Corpus empty — ensure intents.json is loaded and has patterns.")

        logger.info("Generating embeddings for %d corpus entries...", len(corpus))
        embeddings = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
        np.save(EMBEDDING_FILE, embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, INDEX_FILE)
        logger.info("Embeddings and FAISS index saved to disk.")


def load_index():
    """Load FAISS index from file into memory."""
    global index
    with emb_lock:
        if not os.path.exists(INDEX_FILE) or not os.path.exists(EMBEDDING_FILE):
            logger.warning("Embeddings/index files missing; generating now...")
            generate_embeddings_and_index(force=True)
        logger.info("Loading embeddings and FAISS index into memory...")
        embeddings = np.load(EMBEDDING_FILE)
        dim = embeddings.shape[1]
        # read_index returns the index; no need to create dummy first
        index = faiss.read_index(INDEX_FILE)
        logger.info("FAISS index loaded (dim=%d).", dim)


@app.on_event("startup")
def startup_event():
    """
    Called at application start. Loads intents, model, embeddings & index.
    This guarantees one-time model & index load for the lifetime of the process.
    """
    logger.info("Startup: loading intents and model/index...")
    load_intents()
    try:
        generate_embeddings_and_index(force=False)
        load_index()
    except Exception as e:
        logger.exception("Startup generation/loading failed: %s", e)


@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    query = data.get("query", "")
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(status_code=400, detail="Query is required.")
    query = query.strip()

    # Ensure model/index loaded (if startup failed earlier)
    if model is None:
        logger.info("Model not loaded yet; loading on-demand...")
        generate_embeddings_and_index(force=False)
    if index is None:
        load_index()

    # Encode and search
    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    # search top 3 to be safer and allow fallback logic
    k = 3 if index.ntotal >= 3 else 1
    D, I = index.search(query_emb, k=k)
    best_idx = int(I[0][0])
    best_score = float(D[0][0])

    if best_score < SIMILARITY_THRESHOLD:
        return {
            "query": query,
            "intent": "fallback",
            "response": "I'm not sure about that. Could you rephrase or ask something about ACGCET?",
            "similarity": best_score
        }

    # find tag and choose a random response from that tag's responses
    selected_tag = tags[best_idx] if best_idx < len(tags) else None
    if selected_tag is None:
        return {
            "query": query,
            "intent": "fallback",
            "response": "I couldn't match an intent. Please try rephrasing.",
            "similarity": best_score
        }

    responses_for_tag = tag_to_responses.get(selected_tag, ["Sorry, I don't know."])
    response_text = random.choice(responses_for_tag)
    nav_link = tag_to_link.get(selected_tag)

    return {
        "query": query,
        "intent": selected_tag,
        "response": response_text,
        "similarity": best_score,
        "navigation_link": nav_link
    }


@app.get("/generate_embeddings")
def generate_embeddings_endpoint(force: bool = Query(False, description="Force regeneration")):
    """
    Regenerate embeddings & index from current intents.json.
    Use ?force=true to force regeneration even if files exist.
    This endpoint acquires a lock to avoid concurrent regenerations.
    """
    # reload intents (so you can update intents.json and call this endpoint)
    try:
        load_intents()
        generate_embeddings_and_index(force=force)
        load_index()
        return {"status": "ok", "message": "Embeddings and index regenerated and loaded.", "force": bool(force)}
    except Exception as e:
        logger.exception("Failed to generate embeddings: %s", e)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/")
def home():
    return {"message": "ACGCET Chatbot (production-ready) is running 🚀"}

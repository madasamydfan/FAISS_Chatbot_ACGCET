import os
import json
import random
import boto3
import faiss
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# AWS Bedrock token (long-term API key)
token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
if not token:
    raise Exception("AWS Bedrock API key is not set in environment variable")

# FastAPI app
app2 = FastAPI(title="ACGCET Chatbot - AWS Titan", version="2.0")

# Enable CORS (production: restrict origins if needed)
app2.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load intents
with open("intents.json", "r", encoding="utf-8") as f:
    intents_data = json.load(f)

# Prepare corpus, responses, tags, links
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
index = None
EMBEDDING_FILE = "embeddings2.npy"
INDEX_FILE = "faiss_index2.bin"

# AWS Bedrock client
client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
model_id = "amazon.titan-embed-text-v2:0"

def generate_aws_embeddings(text_list):
    """Generate embeddings using AWS Titan model"""
    embeddings = []
    for i, text in enumerate(text_list, start=1):
        print(f"Generating embedding {i}/{len(text_list)}...")
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text})
        )
        result = json.loads(response['body'].read())
        embeddings.append(result["embedding"])
    return np.array(embeddings, dtype=np.float32)

def generate_embeddings_and_index(force: bool = False):
    """Generate embeddings and FAISS index"""
    global index
    if os.path.exists(EMBEDDING_FILE) and os.path.exists(INDEX_FILE) and not force:
        print("✅ Embeddings and FAISS index already exist. Skipping generation.")
        return

    print("⚙️ Generating embeddings using AWS Titan...")
    embeddings = generate_aws_embeddings(corpus)
    np.save(EMBEDDING_FILE, embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    print("✅ Embeddings and FAISS index saved!")

def load_index():
    """Load FAISS index from disk"""
    global index
    if index is None:
        if not os.path.exists(EMBEDDING_FILE) or not os.path.exists(INDEX_FILE):
            print("⚠️ Embeddings or index not found. Generating...")
            generate_embeddings_and_index()
        embeddings = np.load(EMBEDDING_FILE)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index = faiss.read_index(INDEX_FILE)
        print("✅ FAISS index loaded.")

@app2.post("/ask")
async def ask(request: Request):
    data = await request.json()
    query = data.get("query", "").strip()
    if not query:
        return {"error": "Query is required."}

    if index is None:
        load_index()

    query_emb = generate_aws_embeddings([query])
    D, I = index.search(query_emb, k=1)
    best_idx = I[0][0]

    if D[0][0] < 0.5:
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

@app2.get("/generate_embeddings")
def regenerate_embeddings(force: bool = True):
    """Endpoint to regenerate embeddings and FAISS index"""
    generate_embeddings_and_index(force)
    return {"status": "Embeddings and FAISS index generated successfully!"}

@app2.get("/")
def home():
    return {"message": "ACGCET Chatbot with AWS Titan is running 🚀"}

# Run command in production:
# uvicorn app2:app2 --host 0.0.0.0 --port 8000
 #$env:AWS_BEARER_TOKEN_BEDROCK = "Your api key"  
<div align="center">

# 🤖 ACGCET Chatbot

### *Intelligent College Assistant powered by AI*

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Similarity_Search-blue?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A FAISS + Sentence Transformers based chatbot designed for ACGCET students.**  
Answers questions about college facilities, committees, clubs, and general queries using **semantic search**.

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-endpoints) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Adding New Intents](#-adding-new-intents)
- [Dependencies](#-dependencies)
- [Important Notes](#-important-notes)
- [License](#-license)

---

## ✨ Features

- 🔍 **Semantic Search** - Uses sentence embeddings for intelligent query matching
- ⚡ **Fast Response** - FAISS indexing ensures quick similarity search
- 🎯 **High Accuracy** - Sentence Transformers for better understanding
- 🔄 **Easy Updates** - Simple intent management via JSON
- 🌐 **REST API** - FastAPI-powered endpoints for integration
- 📊 **Confidence Scores** - Returns similarity scores for transparency

---

## 📁 Project Structure

```
acgcet-chatbot/
├── 📄 app.py                  # Main FastAPI application
├── 📋 intents.json            # Chatbot intents and patterns
├── 💾 embeddings.npy          # Saved embeddings (auto-generated)
├── 🗂️ faiss_index.bin         # FAISS index file (auto-generated)
├── 📦 requirements.txt        # Python dependencies
├── 🐍 venv/                   # Virtual environment (ignored)
├── 📖 README.md               # Documentation (you are here)
└── 🚫 .gitignore              # Git ignore rules
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/madasamydfan/FAISS_Chatbot_ACGCET.git
cd acgcet-chatbot
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Server

```bash
uvicorn app:app --reload
```

### Step 5: Access the Chatbot

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 💡 Usage

### Method 1: Update Intents

1. Open `intents.json`
2. Add or modify patterns and responses
3. Save the file

### Method 2: Generate Embeddings

After updating intents, regenerate embeddings:

```python
from app import generate_embeddings_and_index
generate_embeddings_and_index(force=True)
```

Or visit the endpoint:
```
http://127.0.0.1:8000/generate_embeddings
```

### Method 3: Query the Chatbot

**POST Request to `/ask`:**

```json
{
    "query": "What facilities does the college provide?"
}
```

**Response:**

```json
{
    "query": "What facilities does the college provide?",
    "intent": "facilities",
    "response": "Our College's department provides fully AC Lab with internet connection, smart classroom, Auditorium, library, canteen...",
    "similarity": 0.89,
    "navigation_link": "https://accet.ac.in/life-at-acgcet"
}
```

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | `GET` | Health check - Test if server is running |
| `/ask` | `POST` | Send a query and get best-matched response |
| `/generate_embeddings` | `GET/POST` | Regenerate embeddings & FAISS index |

### Example: Testing the API

**Using cURL:**
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the college timings?"}'
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/ask",
    json={"query": "What are the college timings?"}
)
print(response.json())
```

---

## ➕ Adding New Intents

1. **Open `intents.json`**

2. **Add a new intent block:**

```json
{
   "tag": "library",
   "patterns": [
      "Library timings",
      "When is library open",
      "Library hours",
      "What time does library open"
   ],
   "responses": [
      "The library is open from 8 AM to 6 PM on weekdays."
   ],
   "context_set": ""
}
```

3. **Regenerate embeddings:**

```python
from app import generate_embeddings_and_index
generate_embeddings_and_index(force=True)
```

Or simply visit: `http://127.0.0.1:8000/generate_embeddings`

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| **fastapi** | Modern web framework for building APIs |
| **uvicorn** | ASGI server to run FastAPI |
| **sentence-transformers** | Generate semantic embeddings from text |
| **faiss-cpu** | Efficient similarity search and clustering |
| **numpy** | Numerical operations and array handling |

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## ⚠️ Important Notes

- 🔄 **Always regenerate embeddings** after editing `intents.json`
- 🚫 `embeddings.npy` and `faiss_index.bin` are **not tracked in git**
- 🎯 Similarity threshold is set to **0.3** (queries below may return fallback)
- 💾 Embeddings are **cached** to improve performance
- 🔧 Use `force=True` to regenerate embeddings manually

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 ACGCET Chatbot Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 👥 Credits

Developed and maintained by **Computer Science and Engineering Students** at ACGCET.

---

<div align="center">

### 🌟 Star this repo if you find it useful!

Made with ❤️ for ACGCET Students

[Report Bug](https://github.com/madasamydfan/FAISS_Chatbot_ACGCET/issues) • [Request Feature](https://github.com/madasamydfan/FAISS_Chatbot_ACGCET/issues)

</div>

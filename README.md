# 📄 AI-Powered PDF Question Answering (RAG)

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask natural language questions.  
The system retrieves relevant content using semantic search (FAISS + embeddings) and generates answers with open-source LLMs.

Built with Streamlit for a simple interactive interface.

---

## ✨ Features

- PDF text extraction
- Intelligent text chunking with overlap
- Semantic similarity search using FAISS
- Embedding-based retrieval
- AI-generated answers via open-source LLMs
- Model selection (TinyLlama / Phi-3)
- GPU acceleration (if available)
- Chat-style UI

---

## 🧠 How It Works

This project follows a standard Retrieval-Augmented Generation pipeline:

1. **PDF Processing** → Extract text from uploaded PDF  
2. **Chunking** → Split text into overlapping segments  
3. **Embeddings** → Convert chunks into dense vectors  
4. **Indexing** → Store vectors in FAISS  
5. **Retrieval** → Find relevant chunks for a query  
6. **Generation** → LLM generates answer using retrieved context  

High-level flow:

PDF → Text → Chunks → Embeddings → FAISS Index  
Question → Embedding → Similarity Search → Context → LLM → Answer

---

## 🛠 Tech Stack

**Frontend / UI**

- Streamlit

**Retrieval & Embeddings**

- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS

**LLM Inference**

- Hugging Face Transformers
- TinyLlama / Phi-3

**Utilities**

- PyPDF
- NumPy
- PyTorch

---
## ▶️ Usage

1. Launch the Streamlit app
2. Load an LLM model
3. Upload a PDF
4. Process PDF
5. Ask questions

---

## 🤖 Supported Models

### TinyLlama-1.1B-Chat

- Faster
- Lightweight
- Lower answer quality

### Phi-3-mini-4k-instruct

- Slower
- Better reasoning
- Higher answer quality

**Note:** GPU recommended for smoother inference.

---

## ⚠️ Limitations

- No OCR support
- Large PDFs may be slow
- CPU inference is slower
---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo

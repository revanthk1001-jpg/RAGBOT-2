# Enterprise RAG Chatbot

An enterprise-grade Retrieval-Augmented Generation (RAG) chatbot built with **FastAPI** (backend) and **Streamlit** (frontend). 

This project allows users to ingest PDF documents, store their contents in a vector database, and chat with the document using state-of-the-art Large Language Models (LLMs). It includes built-in evaluation metrics to score the model's responses and observability tracing to monitor performance in production.

## Features

- **Frontend**: Clean and interactive chat UI built with Streamlit.
- **Backend API**: High-performance REST API powered by FastAPI.
- **LLM Engine**: Integrates local open-source models like **Qwen/Qwen2.5-1.5B-Instruct** for generation and **BAAI/bge-small-en-v1.5** for embeddings via Hugging Face and LangChain.
- **Vector Database**: Uses **FAISS** for fast, local vector storage and semantic retrieval.
- **Evaluation**: Automatically evaluates each RAG response in real-time for **Faithfulness** and **Answer Relevancy** using **DeepEval**.
- **Observability**: Complete end-to-end tracing and callback management using **Langfuse**.

## Prerequisites

- Python 3.9+
- Enough RAM/VRAM to load the local HuggingFace models
- Langfuse Cloud API Keys (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`)

## Installation

1. Clone the repository and navigate into the directory.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Mac/Linux
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory and add your API keys:
   ```env
   LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
   LANGFUSE_SECRET_KEY=your_langfuse_secret_key
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

## Usage

You need to run both the FastAPI backend and the Streamlit frontend simultaneously.

### 1. Start the FastAPI Backend
Open a terminal, activate your virtual environment, and run:
```bash
uvicorn main:app --reload
```
The API will be available at `http://127.0.0.1:8001`. You can view the API documentation at `http://127.0.0.1:8001/docs`.

### 2. Start the Streamlit Frontend
Open a second terminal, activate your virtual environment, and run:
```bash
streamlit run streamlit_app.py
```
This will open the web interface in your default browser.

### 3. Interact
- Upload a PDF using the sidebar.
- Wait for the "Processing & Embedding..." step to complete.
- Ask questions about the document in the chat interface!

## Metrics and Observability

- **DeepEval Metrics**: Every response is scored for Faithfulness and Relevancy. These scores and the total latency are printed in the terminal where your backend is running.
- **Langfuse Tracing**: Every step of the LangChain pipeline (from prompt construction to final output) is automatically traced. Check your Langfuse dashboard for detailed token usage and latency breakdowns.

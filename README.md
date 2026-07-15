# Medical Chatbot

A Streamlit-based medical question answering app that retrieves relevant information from a FAISS vector store and responds to user questions using a retrieval-first workflow.

## Project Overview

This project demonstrates a lightweight RAG-style chatbot for medical information. It uses document embeddings, vector search, and a simple answer-generation flow to provide context-based responses from a curated medical knowledge base.

## Features

- Conversational Streamlit interface
- Retrieval-based answers from medical documents
- FAISS vector search over embedded content
- Docker support for easy deployment
- Environment-based configuration for Gemini and Hugging Face settings
- Simple, modular Python structure for experimentation

## Architecture

1. Medical documents are embedded and stored in a FAISS vector database.
2. A user question is sent to the app through Streamlit.
3. The app retrieves the most relevant document chunks.
4. The response is constructed from those retrieved chunks and shown in the chat UI.

## Tech Stack

- Python 3.13
- Streamlit
- LangChain
- FAISS
- Hugging Face Transformers / sentence-transformers
- Docker
- Pipenv

## Screenshots

Add a screenshot of the Streamlit UI here once available.

## Local Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd medical-chatbot-main
```

### 2. Create and activate a virtual environment with Pipenv

```bash
pipenv install
pipenv shell
```

### 3. Run the app

```bash
streamlit run medibot.py
```

## Docker Setup

### Build the image

```bash
docker build -t medical-chatbot .
```

### Run the container

```bash
docker run -p 8501:8501 medical-chatbot
```

Then open http://localhost:8501 in your browser.

## Environment Variables

Create a `.env` file in the project root with the following:

```env
HF_TOKEN=your_huggingface_token_here
GEMINI_API_KEY=your_gemini_api_key_here
```

The app uses Gemini 2.5 Flash for answer generation and Hugging Face embeddings for retrieval. The app also uses Hugging Face environment variables for offline/cache behavior. These are set in the Dockerfile and app startup code.

## Folder Structure

```text
medical-chatbot-main/
├── medibot.py               # Main Streamlit app
├── create_memory_for_llm.py # Embedding/vectorstore creation helpers
├── connect_memory_with_llm.py
├── data/                    # Source documents or sample content
├── docs/                    # Documentation assets
├── vectorstore/             # FAISS index files
├── Dockerfile               # Container definition
├── requirements.txt         # Python dependencies
├── Pipfile                  # Pipenv environment config
├── .env.example             # Example environment config
└── README.md                # Project overview
```

## Limitations

- The current response quality depends on the quality and coverage of the embedded documents.
- The app is best suited for informational, document-grounded questions rather than highly nuanced medical advice.
- The current implementation is lightweight and intended for demo/prototype use.

## Future Improvements

- Add a stronger LLM backend for improved answer quality
- Improve retrieval ranking and chunking strategy
- Add support for more document formats and ingestion pipelines
- Add authentication and user management for production use
- Improve UI polish and add screenshots/demo assets


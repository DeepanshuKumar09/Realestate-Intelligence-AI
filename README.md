# Real Estate Intelligence AI

An AI-powered real estate intelligence assistant that combines web scraping, Retrieval-Augmented Generation (RAG), semantic search, and conversational AI to provide insights from DarGlobal and Wasalt website data.

## Overview

Real Estate Intelligence AI is a conversational knowledge assistant designed to retrieve and interpret real estate information from two sources:

- **DarGlobal:** Real estate developments, projects, company information, and market-related content.
- **Wasalt:** Property listings and real estate information, including commercial rental listings.

The application uses a RAG architecture to retrieve relevant website content from a persistent vector database and generate context-grounded answers using an LLM.

## Key Features

- **Multi-source Intelligence:** Retrieves information from DarGlobal and Wasalt.
- **Automated Web Scraping:** Collects website content using Requests and BeautifulSoup.
- **Semantic Search:** Uses Sentence Transformers embeddings and ChromaDB.
- **Retrieval-Augmented Generation:** Generates answers using retrieved website documents.
- **Conversational Memory:** Supports contextual follow-up questions.
- **Source-aware Retrieval:** Prioritizes the appropriate website when the user specifies DarGlobal or Wasalt.
- **Streaming Responses:** Displays generated answers progressively.
- **REST API:** Exposes chatbot functionality through FastAPI.
- **Interactive Interface:** Provides a conversational Streamlit application.
- **Containerization:** Supports Docker-based execution.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Web Scraping | Requests, BeautifulSoup, lxml |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| LLM Integration | OpenRouter |
| RAG Framework | LangChain |
| Backend | FastAPI |
| Frontend | Streamlit |
| Containerization | Docker |
| Configuration | Python dotenv |

## System Architecture

```text
                  User
                   |
                   v
              Streamlit UI
                   |
                   v
                FastAPI
                   |
                   v
          Question Processing
                   |
                   v
             Source Detection
                   |
                   v
             Query Rewriting
                   |
                   v
          Sentence Embeddings
                   |
                   v
                ChromaDB
                   |
                   v
          Relevant Documents
                   |
                   v
            Context Assembly
                   |
                   v
             OpenRouter LLM
                   |
                   v
          Streaming Response
                   |
                   v
                  User
```

## RAG Pipeline

### 1. Data Collection
Website content is collected from DarGlobal and Wasalt using custom Python scraping modules.

### 2. Data Processing
Collected content is cleaned and transformed into documents suitable for semantic retrieval.

### 3. Embedding Generation
The `sentence-transformers/all-MiniLM-L6-v2` model converts documents into 384-dimensional vector embeddings.

### 4. Vector Storage
ChromaDB stores document embeddings and associated metadata, including source, title, and URL.

### 5. Retrieval
User questions are converted into semantic queries. Relevant documents are retrieved from ChromaDB, with source filtering when appropriate.

### 6. Answer Generation
Retrieved context is passed to an LLM through LangChain and OpenRouter. The model generates responses grounded in the retrieved website content.

## Project Structure

```text
Realestate-Intelligence-AI/
│
├── app/
│   ├── api/
│   │   └── main.py
│   ├── core/
│   │   └── config.py
│   ├── frontend/
│   │   └── streamlit.py
│   ├── rag/
│   │   ├── embedding.py
│   │   ├── vector_store.py
│   │   └── chain.py
|   |   |__ Preprocessing.py
│   └── scraper/
│       ├── utils.py
│       ├── darglobal.py
│       └── wasalt.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── chroma_db/
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-lock.txt
├── .dockerignore
└── .gitignore
```

## Installation and Setup

### Prerequisites

- Python 3.11
- Docker Desktop (optional)
- OpenRouter API key

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Realestate-Intelligence-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv myvenv
```

Activate on Windows:

```powershell
myvenv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openrouter/free
```

Never commit your `.env` file or expose API credentials.

### 5. Run the Application

Start the FastAPI backend:

```bash
uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

In a separate terminal:

```bash
streamlit run app/frontend/streamlit.py
```

Open the Streamlit URL displayed in your terminal.

## Docker Deployment

Build the Docker image:

```bash
docker build -t real-estate-intelligence-ai .
```

Run the container:

```bash
docker run -d --name real-estate-ai -p 8501:8501 --env-file .env real-estate-intelligence-ai
```

Open:

```text
http://localhost:8501
```

Stop the container:

```bash
docker stop real-estate-ai
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Application health check |
| `/chat` | POST | Conversational RAG endpoint |

The chat endpoint streams responses using NDJSON events.

API documentation:

```text
http://localhost:8000/docs
```

## Knowledge Sources

The application uses publicly accessible website content collected from:

- DarGlobal
- Wasalt

Answers depend on the availability, quality, and freshness of the collected website data. Property listings and other details may change over time.

## Future Improvements

- Scheduled website data updates.
- Improved retrieval ranking and evaluation.
- Advanced hybrid search.
- User authentication and session management.
- Production monitoring and observability.
- Integration of additional real estate data sources.

## Project Information

Built with Python, LangChain, ChromaDB, FastAPI, Streamlit, and OpenRouter.

import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

from app.rag.embedding import get_embedding_model


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BASE_DIR / ".env")

VECTOR_DB_DIR = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "real_estate_knowledge"


# --------------------------------------------------
# LLM INITIALIZATION
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_llm():

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing")

    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_retries=2,
        timeout=120,
        streaming=True
    )


# --------------------------------------------------
# VECTOR DATABASE
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_vector_store():

    embedding_model = get_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(VECTOR_DB_DIR)
    )


# --------------------------------------------------
# SOURCE DETECTION
# --------------------------------------------------

def detect_source(question: str):

    """
    Detect the requested knowledge source from the current question.
    """

    text = question.lower()

    if "wasalt" in text:
        return "Wasalt"

    if "darglobal" in text or "dar global" in text:
        return "DarGlobal"

    return None


# --------------------------------------------------
# CONVERSATIONAL QUESTION REWRITING
# --------------------------------------------------

def rewrite_question(question: str, history: list):

    if not history:
        return question

    # Explicit company questions should not inherit
    # the previous company's context.

    if detect_source(question):
        return question

    conversation = []

    for message in history[-6:]:

        role = message.get("role", "user")
        content = message.get("content", "")

        if role in ["user", "assistant"] and content:
            conversation.append(f"{role.upper()}: {content}")

    history_text = "\n".join(conversation)

    prompt = f"""
You are a conversational search query rewriting assistant.

Convert the latest user question into a standalone search query.

Rules:
- Resolve references such as it, they, their and those properties.
- Preserve the user's original intent.
- Do not answer the question.
- Do not introduce unsupported facts.
- Return only the rewritten query.

CONVERSATION:
{history_text}

LATEST QUESTION:
{question}

STANDALONE SEARCH QUERY:
"""

    response = get_llm().invoke(prompt)

    rewritten = response.content.strip()

    return rewritten if rewritten else question


# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

def retrieve_documents(search_query: str, source: str = None):

    vector_store = get_vector_store()

    if source:

        # Restrict retrieval to the selected website.

        docs = vector_store.similarity_search(
            search_query,
            k=8,
            filter={"source": source}
        )

        return docs

    # Search both websites when no source is specified.

    return vector_store.similarity_search(
        search_query,
        k=8
    )


# --------------------------------------------------
# RAG RETRIEVAL + STREAMING ANSWER
# --------------------------------------------------

def get_answer_stream(question: str, history: list = None):

    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    history = history or []

    # STEP 1: Detect source from the current question.

    selected_source = detect_source(question)

    # STEP 2: Rewrite follow-up questions.

    search_query = rewrite_question(
        question,
        history
    )

    # STEP 3: Retrieve relevant documents.

    relevant_docs = retrieve_documents(
        search_query,
        selected_source
    )

    # STEP 4: Handle missing context.

    if not relevant_docs:

        source_name = selected_source or "DarGlobal and Wasalt"

        fallback = (
            f"I couldn't find relevant information about "
            f"{source_name} in the available website data. "
            "Please try asking about a specific project, "
            "location, property or company detail."
        )

        def fallback_stream():
            yield fallback

        return fallback_stream(), []

    # STEP 5: Build retrieved context and source references.

    context_parts = []
    sources = []
    seen_urls = set()

    for doc in relevant_docs:

        metadata = doc.metadata

        title = metadata.get("title", "Unknown")
        source = metadata.get("source", "Unknown")
        url = metadata.get("url", "")

        context_parts.append(
            f"""
WEBSITE: {source}
TITLE: {title}
URL: {url}

DOCUMENT CONTENT:
{doc.page_content}
"""
        )

        if url and url not in seen_urls:

            seen_urls.add(url)

            sources.append({
                "source": source,
                "title": title,
                "url": url
            })

    context = "\n\n".join(context_parts)

    # STEP 6: Prepare conversation history.

    conversation_history = []

    for message in history[-6:]:

        role = message.get("role", "user")
        content = message.get("content", "")

        if role in ["user", "assistant"] and content:
            conversation_history.append(
                f"{role.upper()}: {content}"
            )

    history_text = "\n".join(conversation_history)

    # STEP 7: Generate grounded professional answer.

    answer_prompt = f"""
You are a professional Real Estate Intelligence Analyst.

You answer questions using retrieved information from:
1. DarGlobal
2. Wasalt

SELECTED SOURCE:
{selected_source or "Automatically selected from retrieved documents"}

STRICT KNOWLEDGE RULES:

1. Use ONLY the retrieved website context for factual claims.
2. The retrieved website context is your primary source of truth.
3. Never assume that DarGlobal and Wasalt are the same company.
4. Never replace Wasalt information with DarGlobal information.
5. Never invent property details, prices, amenities or locations.
6. If the retrieved context contains Wasalt documents,
   answer using those documents.
7. If the retrieved context contains DarGlobal documents,
   answer using those documents.
8. If the requested information is unavailable, clearly say so.
9. Conversation history is only for understanding follow-up questions.
   It is NOT a verified factual source.
10. Never use facts from previous assistant answers unless
    supported by the retrieved website context.

FORMATTING RULES:

1. Use professional, natural language.
2. Use meaningful bold headings when appropriate.
3. Present details using concise bullet points.
4. Bold important locations, project names and figures.
5. Avoid repetitive explanations.
6. Never write [Source 1], [Source 2] or similar labels.
7. Do not generate URLs or hyperlinks.
8. Do not include a separate sources list.
9. Answer simple questions directly.
10. Do not claim information is unavailable when it is
    clearly present in the retrieved context.

CONVERSATION HISTORY:
{history_text}

RETRIEVED WEBSITE CONTEXT:
{context}

LATEST USER QUESTION:
{question}

Write a clear, accurate and professional answer in Markdown.
"""

    # STEP 8: Stream the answer.

    llm = get_llm()

    stream = llm.stream(answer_prompt)

    return stream, sources


# --------------------------------------------------
# LOCAL TEST
# --------------------------------------------------

if __name__ == "__main__":

    question = (
        "Find commercial properties for rent "
        "in Al Rakah Al Janubiyah, Saudi Arabia."
    )

    stream, sources = get_answer_stream(question)

    print("\nANSWER:\n")

    for chunk in stream:

        if chunk.content:
            print(chunk.content, end="", flush=True)

    print("\n\nSOURCES:\n")

    for source in sources:

        print("Website:", source["source"])
        print("Title:", source["title"])
        print("URL:", source["url"])
        print()


from dotenv import load_dotenv
from pathlib import Path
import os

from langchain_openai import ChatOpenAI

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.2
)

response = llm.invoke("Explain what RAG means in two sentences.")

print("OpenRouter connection successful!")
print(response.content)

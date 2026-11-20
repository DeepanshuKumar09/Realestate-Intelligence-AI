
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.rag.chain import get_answer_stream


app = FastAPI(
    title="Real Estate Intelligence AI",
    description="Conversational AI assistant for DarGlobal and Wasalt",
    version="2.0.0"
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "Real Estate Intelligence AI"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    try:
        history = [
            message.model_dump()
            for message in request.history
        ]

        stream, sources = get_answer_stream(
            request.question,
            history
        )

        def generate():
            # Send source references first
            yield json.dumps({
                "type": "sources",
                "data": sources
            }) + "\n"

            # Send generated answer token by token
            for chunk in stream:
                if chunk.content:
                    yield json.dumps({
                        "type": "token",
                        "data": chunk.content
                    }) + "\n"

            # Mark response completion
            yield json.dumps({
                "type": "done",
                "data": ""
            }) + "\n"

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

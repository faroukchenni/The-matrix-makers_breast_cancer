from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Optional
import os

from openai import OpenAI

router = APIRouter(prefix="/chat", tags=["chat"])

# Client reads OPENAI_API_KEY from environment
client = OpenAI()

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    # Optional: allow choosing model later if you want
    model: Optional[str] = "gpt-4o-mini"

@router.post("")
def chat(req: ChatRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing on backend.")

    try:
        resp = client.chat.completions.create(
            model=req.model or "gpt-4o-mini",
            messages=[m.model_dump() for m in req.messages],
            temperature=0.4,
        )
        return {
            "reply": resp.choices[0].message.content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

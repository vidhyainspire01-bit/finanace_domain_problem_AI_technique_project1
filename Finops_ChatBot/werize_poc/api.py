from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from controller import get_response
import uvicorn

app = FastAPI(title="WeRize Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str | None = None
    user_input: str

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    text, docs, tok = get_response(req.user_input, history=[])
    return {"reply": text, "docs": docs, "usage": tok}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
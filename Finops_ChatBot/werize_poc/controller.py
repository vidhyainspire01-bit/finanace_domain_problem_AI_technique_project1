# controller.py
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from retriever import Retriever, init_pinecone
load_dotenv()

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
PINE_API = os.getenv("PINECONE_API_KEY")
PINE_ENV = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
INDEX_NAME = os.getenv("PINECONE_INDEX", "werize-chatbot")

client = Anthropic(api_key=ANTHROPIC_KEY)

pine_index = None
if PINE_API:
    try:
        pine_index = init_pinecone(PINE_API, INDEX_NAME)
    except Exception as e:
        print("Pinecone init failed:", e)

retriever = Retriever(pine_index)

SYSTEM_PROMPT = """You are WeRize Assistant, a fintech chatbot. Use retrieved documents for facts, cite sources briefly, never reveal PII, ask for OTP before any transaction."""

# helper to trim history (keep last N user/assistant pairs)
def trim_history(messages, max_pairs=6):
    # messages is list of dicts {role, content}
    if not messages:
        return []
    # keep last 2*max_pairs messages
    return messages[-(max_pairs*2):]

def get_response(user_text, history, max_tokens=250, temperature=0.2):
    # 1) retrieve relevant docs
    docs = retriever.query(user_text, top_k=3)
    context = "\n".join([f"- {d}" for d in docs])
    # 2) build messages for Anthropic: system is top-level, messages only user/assistant
    # build a single user message including context and the question
    assistant_input = f"Context:\n{context}\n\nQuestion: {user_text}"
    messages = [{"role":"user","content":assistant_input}]
    # 3) call Anthropic
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        system=SYSTEM_PROMPT,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    text = response.content[0].text
    # optional: extract usage
    usage = getattr(response, "usage", None)
    token_info = None
    if usage is not None:
        # safe extraction
        i = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
        o = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
        t = getattr(usage, "total_tokens", None) or ( (i or 0) + (o or 0) )
        token_info = {"input_tokens":i, "output_tokens":o, "total_tokens":t}
    return text, docs, token_info

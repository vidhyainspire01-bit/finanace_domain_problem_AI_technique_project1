# ingest.py
import os
from retriever import Retriever, init_pinecone, embed_text
from dotenv import load_dotenv

load_dotenv()
PINE_API = os.getenv("PINECONE_API_KEY")
PINE_ENV = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
INDEX_NAME = os.getenv("PINECONE_INDEX", "werize-chatbot")

pine_index = None
if PINE_API:
    try:
        pine_index = init_pinecone(PINE_API, PINE_ENV, INDEX_NAME)
    except Exception as e:
        print("Pinecone init failed:", e)
ret = Retriever(pine_index)

# Read sample file
with open("data/werize_sample.txt", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]

for i,l in enumerate(lines):
    doc_id = f"doc_{i+1}"
    ret.upsert(doc_id, l)
    print("Upserted:", doc_id)

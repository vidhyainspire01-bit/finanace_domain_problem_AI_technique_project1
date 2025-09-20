# check_pinecone.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pine_key = os.getenv("PINECONE_API_KEY")
pine_env = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")

if not pine_key:
    raise SystemExit("PINECONE_API_KEY not set")

pc = Pinecone(api_key=pine_key)

print("Pinecone client connected.")
print("Indexes:", pc.list_indexes().names())

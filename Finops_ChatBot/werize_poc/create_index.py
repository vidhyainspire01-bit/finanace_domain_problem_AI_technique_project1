# create_index.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pine_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX", "werize-chatbot")

if not pine_key:
    raise SystemExit("PINECONE_API_KEY not set")

pc = Pinecone(api_key=pine_key)

# Check if index already exists
if index_name in pc.list_indexes().names():
    print("Index already exists:", index_name)
else:
    pc.create_index(
        name=index_name,
        dimension=384,          # using MiniLM embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Created index:", index_name)

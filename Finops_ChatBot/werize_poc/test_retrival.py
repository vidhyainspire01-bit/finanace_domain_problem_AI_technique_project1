import os
from dotenv import load_dotenv
load_dotenv()
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

pine_key = os.getenv("PINECONE_API_KEY"); idx = os.getenv("PINECONE_INDEX","werize-chatbot")
pc = Pinecone(api_key=pine_key); index = pc.Index(idx)
m = SentenceTransformer("all-MiniLM-L6-v2")
q = "How can I repay my loan?"
qv = m.encode(q).tolist()
res = index.query(vector=qv, top_k=3, include_metadata=True)
print("Matches:")
for match in res["matches"]:
    s = match.get("score", match.get("match_score"))
    print("-", s, match["metadata"]["content"])
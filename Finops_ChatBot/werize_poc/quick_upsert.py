# quick_upsert.py
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv()
pine_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX", "werize-chatbot")

if not pine_key:
    raise SystemExit("PINECONE_API_KEY not set")

pc = Pinecone(api_key=pine_key)
index = pc.Index(index_name)

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text):
    return model.encode(text).tolist()

docs = [
    ("faq1","WeRize offers flexible loan repayment via UPI, bank transfer and auto-debit. For UPI, use your virtual ID or scan the QR."),
    ("faq2","KYC for customers requires Aadhaar, PAN and one address proof (utility bill or bank statement). PAN is mandatory for loans above ₹50,000."),
    ("faq3","Group insurance claims: submit claim form, ID proof, supporting medical bills; claims processed within 15 working days.")
]

for doc_id, text in docs:
    v = embed_text(text)
    index.upsert([(doc_id, v, {"content": text})])
    print("Upserted", doc_id)
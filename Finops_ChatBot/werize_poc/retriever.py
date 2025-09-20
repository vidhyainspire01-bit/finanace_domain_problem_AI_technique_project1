import os
from sentence_transformers import SentenceTransformer

PINECONE_ENABLED = True
try:
    from pinecone import Pinecone, ServerlessSpec
except Exception:
    PINECONE_ENABLED = False

EMBED_MODEL = "all-MiniLM-L6-v2"
DIM = 384

# Load embedding model once
model = SentenceTransformer(EMBED_MODEL)

def embed_text(text):
    return model.encode(text).tolist()

def init_pinecone(api_key, index_name):
    """
    Initialize Pinecone client and ensure index exists.
    Returns: an Index object (pc.Index(index_name))
    """
    if not api_key:
        raise ValueError("Pinecone API key missing")

    pc = Pinecone(api_key=api_key)

    # create index if needed
    names = pc.list_indexes().names()
    if index_name not in names:
        # create serverless index in a free-tier friendly region
        try:
            pc.create_index(
                name=index_name,
                dimension=DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        except Exception:
            # fallback to gcp region if aws fails
            pc.create_index(
                name=index_name,
                dimension=DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="gcp", region="us-east1")
            )

    return pc.Index(index_name)

class Retriever:
    def __init__(self, pinecone_index=None):
        self.index = pinecone_index
        self.local_store = []  # fallback: list of tuples (id, text, vec)

    def upsert(self, doc_id, text):
        vec = embed_text(text)
        if self.index is not None:
            self.index.upsert([(doc_id, vec, {"content": text})])
        else:
            self.local_store.append((doc_id, text, vec))

    def query(self, query_text, top_k=3):
        qv = embed_text(query_text)
        if self.index is not None:
            res = self.index.query(vector=qv, top_k=top_k, include_metadata=True)
            # defensive: check structure
            matches = res.get("matches") if isinstance(res, dict) else getattr(res, "matches", None)
            if not matches:
                return []
            return [m["metadata"]["content"] for m in matches]
        # Fallback local cosine similarity
        import numpy as np
        def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
        scores = [(cos(qv, v), doc, txt) for doc, txt, v in self.local_store]
        scores.sort(reverse=True)
        return [txt for _,doc,txt in scores[:top_k]]
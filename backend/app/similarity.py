from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

# Load data
with open("sample_data/sample_data.json") as f:
    data = json.load(f)

corpus = [entry["user_story"] for entry in data]

model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = model.encode(corpus, normalize_embeddings=True)

index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
index.add(corpus_embeddings)

def get_similar_examples(query, k=3):
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_embedding, k)
    return [data[i] for i in indices[0]]

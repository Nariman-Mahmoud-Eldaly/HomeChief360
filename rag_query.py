# rag_query.py
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Optionally use OpenAI if you set the env var OPENAI_API_KEY
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY", None))
if USE_OPENAI:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# For local generation (preferred offline)
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

INDEX_FILE = "recipes_faiss.index"
MAPPING_FILE = "id_to_meta.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

LOCAL_GEN_MODEL = os.getenv("LOCAL_GEN_MODEL", "./flan-t5-small")  

# Load embedding model & FAISS index
print("Loading embedding model:", EMBEDDING_MODEL)
embed_model = SentenceTransformer(EMBEDDING_MODEL)

print("Loading FAISS index:", INDEX_FILE)
index = faiss.read_index(INDEX_FILE)

print("Loading id->meta mapping:", MAPPING_FILE)
with open(MAPPING_FILE, "rb") as f:
    id_to_meta = pickle.load(f)

# Setup local generator
print("Loading local generation model:", LOCAL_GEN_MODEL)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_GEN_MODEL)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_GEN_MODEL)
generator = pipeline("text2text-generation", model=gen_model, tokenizer=tokenizer, device=-1)  # CPU device

def retrieve(query: str, top_k:int = 3):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, top_k)
    results = []
    for idx in indices[0]:
        meta = id_to_meta.get(int(idx))
        if meta:
            results.append(meta)
    return results

def generate_with_local_model(query: str, retrieved: list, max_length: int = 200):
    # Build context prompt
    context = ""
    for r in retrieved:
        context += f"Title: {r['title']}\nIngredients: {r['ingredients']}\nDirections: {r['directions']}\n\n"
    prompt = f"Using the recipes below, produce a concise, step-by-step recipe that answers: {query}\n\nRecipes:\n{context}\nAnswer:"
    out = generator(prompt, max_length=max_length, do_sample=False)[0]['generated_text']
    return out

def generate_with_openai(query: str, retrieved: list):
    context = ""
    for r in retrieved:
        context += f"Title: {r['title']}\nIngredients: {r['ingredients']}\nDirections: {r['directions']}\n\n"
    prompt = f"You are a helpful cooking assistant. The user asked: {query}\n\nHere are some related recipes:\n{context}\nPlease write a concise step-by-step recipe suggestion for the user."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        max_tokens=400
    )
    return response.choices[0].message.content




def rag_answer(query: str, top_k: int = 3, prefer_local: bool = True):
    retrieved = retrieve(query, top_k=top_k)
    if USE_OPENAI and not prefer_local:
        return generate_with_openai(query, retrieved)
    else:
        return generate_with_local_model(query, retrieved)

if __name__ == "__main__":
    q = "Give me a simple vegan pasta recipe with tomato"
    print("Query:", q)
    res = rag_answer(q, top_k=3, prefer_local=True)
    print("Generated answer:\n", res)

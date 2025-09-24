# build_index.py
import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from tqdm.auto import tqdm

CLEAN_PARQUET = "recipes_clean.parquet"   
SAMPLE_CSV = "recipes_sample_clean.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 512                           
INDEX_FILE = "recipes_faiss.index"
MAPPING_FILE = "id_to_meta.pkl"
EMBEDDINGS_NPY = "recipe_embeddings.npy"   

def load_data(use_sample=False):
    if use_sample and Path(SAMPLE_CSV).exists():
        df = pd.read_csv(SAMPLE_CSV)
    elif Path(CLEAN_PARQUET).exists():
        df = pd.read_parquet(CLEAN_PARQUET)
    else:
        raise FileNotFoundError("No cleaned dataset found. Run prepare_data.py first.")
    return df.reset_index(drop=True)

def make_corpus_text(row):
    # Create a single text per recipe for embedding
    return f"{row['title']} || Ingredients: {row['ingredients']} || Directions: {row['directions']}"

def main(use_sample=True):
    df = load_data(use_sample=use_sample)
    print("Loaded rows:", len(df))

    # Build corpus
    texts = [make_corpus_text(r) for _, r in df.iterrows()]

    # Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)
    model.max_seq_length = 512

    # Compute embeddings in batches
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding batches"):
        batch_texts = texts[i: i + BATCH_SIZE]
        emb = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(emb)
    embeddings = np.vstack(all_embeddings).astype("float32")
    print("Embeddings shape:", embeddings.shape)

    # Save embeddings numpy (optional, may be large)
    np.save(EMBEDDINGS_NPY, embeddings)
    print("Saved embeddings to", EMBEDDINGS_NPY)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print("Index ntotal:", index.ntotal)

    # Save index
    faiss.write_index(index, INDEX_FILE)
    print("Saved FAISS index to", INDEX_FILE)

    # Save mapping (index -> metadata)
    id_to_meta = {}
    for i, row in df.iterrows():
        id_to_meta[i] = {"title": row["title"], "ingredients": row["ingredients"], "directions": row["directions"]}
    with open(MAPPING_FILE, "wb") as f:
        pickle.dump(id_to_meta, f)
    print("Saved mapping to", MAPPING_FILE)

if __name__ == "__main__":
    main(use_sample=True)

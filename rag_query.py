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

def filter_recipes_by_preferences(recipes: list, user_preferences: dict) -> list:
   
    if not user_preferences or not recipes:
        return recipes
    
    filtered_recipes = []
    
    for recipe in recipes:
        ingredients_text = ""
        if isinstance(recipe['ingredients'], list):
            ingredients_text = " ".join(recipe['ingredients']).lower()
        else:
            ingredients_text = str(recipe['ingredients']).lower()
        
        score = 0
        
        favorite_ingredients = user_preferences.get('favorite_ingredients', [])
        if favorite_ingredients:
            favorite_count = sum(1 for fav in favorite_ingredients if fav and fav.lower() in ingredients_text)
            if favorite_count > 0:
                score += favorite_count * 2  
            else:
                score -= 1
        
        ingredients_to_avoid = user_preferences.get('ingredients_to_avoid', [])
        avoid_violation = False
        if ingredients_to_avoid:
            avoid_count = sum(1 for avoid in ingredients_to_avoid if avoid and avoid.lower() in ingredients_text)
            if avoid_count > 0:
                avoid_violation = True
                score -= avoid_count * 10  
        allergies = user_preferences.get('allergies', [])
        allergy_violation = False
        if allergies:
            allergy_count = sum(1 for allergy in allergies if allergy and allergy.lower() in ingredients_text)
            if allergy_count > 0:
                allergy_violation = True
                score -= allergy_count * 20  
        
        dietary_restrictions = user_preferences.get('dietary_restrictions', [])
        if dietary_restrictions:
            if 'vegan' in dietary_restrictions and any(non_vegan in ingredients_text for non_vegan in ['milk', 'cheese', 'butter', 'egg', 'meat', 'chicken', 'fish']):
                score -= 15
            if 'vegetarian' in dietary_restrictions and any(non_veg in ingredients_text for non_veg in ['meat', 'chicken', 'fish']):
                score -= 15
        
      
        if not allergy_violation and score > -5:  
            filtered_recipes.append((recipe, score))
    
    filtered_recipes.sort(key=lambda x: x[1], reverse=True)
    
    return [recipe for recipe, score in filtered_recipes]

def generate_with_local_model(query: str, retrieved: list, user_preferences: dict = None, max_new_tokens: int = 256):
    if not retrieved:
        backup_retrieved = retrieve(query, top_k=3)
        if not backup_retrieved:
            return "I couldn't find any recipes matching your preferences. Please try broadening your search criteria."
        
        context = ""
        for r in backup_retrieved:
            context += f"Title: {r['title']}\nIngredients: {r['ingredients']}\nDirections: {r['directions']}\n\n"
        
        prompt = f"Based on these recipes, suggest alternatives that might work for the user's query: {query}\n\nRecipes:\n{context}\nSuggestion:"
    else:
        # Build context prompt with user preferences
        context = ""
        for r in retrieved:
            context += f"Title: {r['title']}\nIngredients: {r['ingredients']}\nDirections: {r['directions']}\n\n"
        
        # Add user preferences to the prompt
        preferences_text = ""
        if user_preferences:
            preferences_text = "\nUser Preferences:\n"
            if user_preferences.get('dietary_restrictions'):
                preferences_text += f"- Dietary restrictions: {', '.join(user_preferences['dietary_restrictions'])}\n"
            if user_preferences.get('allergies'):
                preferences_text += f"- Allergies: {', '.join(user_preferences['allergies'])}\n"
            
            if user_preferences.get('ingredients_to_avoid'):
                preferences_text += f"- Ingredients to avoid: {', '.join(user_preferences['ingredients_to_avoid'])}\n"
            if user_preferences.get('favorite_ingredients'):
                preferences_text += f"- Favorite ingredients: {', '.join(user_preferences['favorite_ingredients'])}\n"
           
        
        prompt = f"Using the recipes below and considering the user preferences, produce a personalized, concise, step-by-step recipe that answers: {query}\n\n{preferences_text}\nRecipes:\n{context}\nPersonalized Recipe Answer:"
    
    out = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]['generated_text']
    return out

def generate_with_openai(query: str, retrieved: list, user_preferences: dict = None):
    if not retrieved:
        backup_retrieved = retrieve(query, top_k=3)
        if not backup_retrieved:
            return "I couldn't find any recipes matching your preferences. Please try broadening your search criteria."
        
        context = ""
        for r in backup_retrieved:
            context += f"Title: {r['title']}\nIngredients: {r['ingredients']}\nDirections: {r['directions']}\n\n"
        
        prompt = f"The user has specific preferences but I couldn't find perfect matches. Based on these recipes, suggest alternatives for: {query}\n\nRecipes:\n{context}\nHelpful Suggestion:"
    else:
        context = ""
        for r in retrieved:
            context += f"Title: {r['title']}\nIngredients: {r['ingredients']}\nDirections: {r['directions']}\n\n"
        
        # Add user preferences to the prompt
        preferences_text = ""
        if user_preferences:
            preferences_text = "\nUser Preferences:\n"
            if user_preferences.get('dietary_restrictions'):
                preferences_text += f"- Dietary restrictions: {', '.join(user_preferences['dietary_restrictions'])}\n"
            if user_preferences.get('allergies'):
                preferences_text += f"- Allergies: {', '.join(user_preferences['allergies'])}\n"
           
            if user_preferences.get('ingredients_to_avoid'):
                preferences_text += f"- Ingredients to avoid: {', '.join(user_preferences['ingredients_to_avoid'])}\n"
            if user_preferences.get('favorite_ingredients'):
                preferences_text += f"- Favorite ingredients: {', '.join(user_preferences['favorite_ingredients'])}\n"
            
        
        prompt = f"You are a helpful cooking assistant that considers user preferences. The user asked: {query}\n\n{preferences_text}\nHere are some related recipes:\n{context}\nPlease write a personalized, concise step-by-step recipe suggestion that considers the user's preferences and restrictions."
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        max_tokens=400
    )
    return response.choices[0].message.content

def rag_answer(query: str, top_k: int = 3, prefer_local: bool = True, user_preferences: dict = None):
    retrieved = retrieve(query, top_k=top_k * 3)
    
    if user_preferences:
        retrieved = filter_recipes_by_preferences(retrieved, user_preferences)
    
    retrieved = retrieved[:top_k]
    
    if USE_OPENAI and not prefer_local:
        return generate_with_openai(query, retrieved, user_preferences)
    else:
        return generate_with_local_model(query, retrieved, user_preferences)

if __name__ == "__main__":
    q = "Give me a simple vegan pasta recipe with tomato"
    print("Query:", q)
    res = rag_answer(q, top_k=3, prefer_local=True)
    print("Generated answer:\n", res)
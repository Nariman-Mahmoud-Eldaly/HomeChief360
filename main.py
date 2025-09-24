#main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import json
from rag_query import rag_answer, retrieve
from transformers import pipeline
app = FastAPI()

def clean_text_field(field):
    if isinstance(field, str):
        try:
            return json.loads(field)  
        except:
            return [field]  
    return field


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    prefer_local: bool = True

class SmartRequest(BaseModel):
    ingredients: Optional[List[str]] = None
    recipe: Optional[str] = None
    top_k: int = 3



class ChatRequest(BaseModel):
    text: str
    top_k: int = 3
    prefer_local: bool = True

#chat 
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_text = req.text.lower()

    if "recipe of" in user_text or "make" in user_text or "how to" in user_text:
        query_type = "recipe_name"
    else:
        query_type = "ingredients"

    if query_type == "recipe_name":
        retrieved = retrieve(req.text, top_k=req.top_k)
    else:
        ingredients = [i.strip() for i in req.text.replace("and",",").split(",")]
        query = "Recipe using: " + ", ".join(ingredients)
        retrieved = retrieve(query, top_k=req.top_k)

    

    return {
        "user_query": req.text,
        "query_type": query_type,
        "retrieved_recipes": [
            {
                "title": doc["title"],
                "ingredients": clean_text_field(doc["ingredients"]),
                "directions": clean_text_field(doc["directions"]),
            } for doc in retrieved
        ],
        
    }

@app.get("/")
def read_root():
    return {"message": "Recipe RAG API is running!"}


# 2) Smart endpoint (ingredients -> recipe OR recipe -> ingredients)
@app.post("/ask")
def smart_ask(req: SmartRequest):
    if req.ingredients:
        query = "Recipe using: " + ", ".join(req.ingredients)
        retrieved = retrieve(query, top_k=req.top_k)
        return {
            "mode": "ingredients_to_recipe",
            "ingredients_given": req.ingredients,
            "possible_recipes": [
                {
                    "title": doc["title"],
                    "ingredients": clean_text_field(doc["ingredients"]),
                    "directions": clean_text_field(doc["directions"]),
                }
                for doc in retrieved
            ],
        }
    elif req.recipe:
        query = "Ingredients for recipe: " + req.recipe
        retrieved = retrieve(query, top_k=req.top_k)
        return {
            "mode": "recipe_to_ingredients",
            "recipe": req.recipe,
            "found_recipes": [
                {
                    "title": doc["title"],
                    "ingredients": clean_text_field(doc["ingredients"]),
                    "directions": clean_text_field(doc["directions"]),
                }
                for doc in retrieved
            ],
        }
    else:
        return {"Please provide either ingredients or recipe"}
































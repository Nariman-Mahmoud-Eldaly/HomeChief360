# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
from rag_query import rag_answer, retrieve, filter_recipes_by_preferences
from transformers import pipeline
app = FastAPI()

def clean_text_field(field):
    if isinstance(field, str):
        try:
            return json.loads(field)  
        except:
            return [field]  
    return field

class UserPreferences(BaseModel):
    dietary_restrictions: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    ingredients_to_avoid: Optional[List[str]] = None
    favorite_ingredients: Optional[List[str]] = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    prefer_local: bool = True

class SmartRequest(BaseModel):
    ingredients: Optional[List[str]] = None
    recipe: Optional[str] = None
    top_k: int = 3
    user_preferences: Optional[UserPreferences] = None

class ChatRequest(BaseModel):
    text: str
    top_k: int = 3
    prefer_local: bool = True
    user_preferences: Optional[UserPreferences] = None

#chat endpoint 
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_text = req.text.lower()

    if "recipe of" in user_text or "make" in user_text or "how to" in user_text:
        query_type = "recipe_name"
    else:
        query_type = "ingredients"

    if query_type == "recipe_name":
        retrieved = retrieve(req.text, top_k=req.top_k * 3)
    else:
        ingredients = [i.strip() for i in req.text.replace("and",",").split(",")]
        query = "Recipe using: " + ", ".join(ingredients)
        retrieved = retrieve(query, top_k=req.top_k * 3)

    # preferences
    filtered_recipes = retrieved
    if req.user_preferences:
        filtered_recipes = filter_recipes_by_preferences(retrieved, req.user_preferences.dict())
        
        filtered_recipes = filtered_recipes[:req.top_k]

    response_data = {
        "user_query": req.text,
        "query_type": query_type,
        "retrieved_recipes": [
            {
                "title": doc["title"],
                "ingredients": clean_text_field(doc["ingredients"]),
                "directions": clean_text_field(doc["directions"]),
            } for doc in filtered_recipes
        ],
    }

    # Add user preferences if available
    if req.user_preferences:
        response_data["user_preferences"] = req.user_preferences.dict()

    return response_data

@app.get("/")
def read_root():
    return {"message": "Recipe RAG API is running!"}

@app.post("/ask")
def smart_ask(req: SmartRequest):
    if req.ingredients:
        query = "Recipe using: " + ", ".join(req.ingredients)
        retrieved = retrieve(query, top_k=req.top_k * 3)
        
        filtered_recipes = retrieved
        if req.user_preferences:
            filtered_recipes = filter_recipes_by_preferences(retrieved, req.user_preferences.dict())
            filtered_recipes = filtered_recipes[:req.top_k]

        response_data = {
            "mode": "ingredients_to_recipe",
            "ingredients_given": req.ingredients,
            "possible_recipes": [
                {
                    "title": doc["title"],
                    "ingredients": clean_text_field(doc["ingredients"]),
                    "directions": clean_text_field(doc["directions"]),
                }
                for doc in filtered_recipes
            ],
        }

        # Add user preferences if available
        if req.user_preferences:
            response_data["user_preferences"] = req.user_preferences.dict()

        return response_data

    elif req.recipe:
        query = "Ingredients for recipe: " + req.recipe
        retrieved = retrieve(query, top_k=req.top_k * 3)
        
        filtered_recipes = retrieved
        if req.user_preferences:
            filtered_recipes = filter_recipes_by_preferences(retrieved, req.user_preferences.dict())
            filtered_recipes = filtered_recipes[:req.top_k]

        response_data = {
            "mode": "recipe_to_ingredients",
            "recipe": req.recipe,
            "found_recipes": [
                {
                    "title": doc["title"],
                    "ingredients": clean_text_field(doc["ingredients"]),
                    "directions": clean_text_field(doc["directions"]),
                }
                for doc in filtered_recipes
            ],
        }

        # Add user preferences if available
        if req.user_preferences:
            response_data["user_preferences"] = req.user_preferences.dict()

        return response_data
    else:
        return {"Please provide either ingredients or recipe"}
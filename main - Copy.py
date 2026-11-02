import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

# Load variables from your local .env file
load_dotenv()

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

@app.post("/api/v1/chat")
async def handle_chat(payload: ChatRequest):
    # Retrieve the free Gemini key you saved in Step 2
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing in your .env file.")
    
    try:
        # Initialize the official Google GenAI client
        client = genai.Client(api_key=api_key)
        
        # Call the fast and free Gemini 2.5 Flash model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=payload.prompt,
        )
        
        return {"status": "success", "response": response.text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

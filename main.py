from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

# Allow requests from frontend
origins = os.getenv("ALLOWED_ORIGINS", "").split(",")  
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/chat")
async def chat(query: str):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        chat_session = model.start_chat(history=[
            {"role": "user", "parts": [
                "You are NeuraChat, an AI assistant that responds only in plain text. "
                "Do NOT generate images, videos, or files."
            ]}
        ])

        # Send user message
        response = chat_session.send_message(query)

        return {"response": response.text}
    except Exception as e:
        return {"response": f"⚠️ Error: {str(e)}"}




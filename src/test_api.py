import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
# Test with a single word to see if the 404 persists
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector = embeddings.embed_query("test")
    print("✅ Connection Successful! The model was found.")
except Exception as e:
    print(f"❌ Connection Failed: {e}")
import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

models = genai.list_models()
for m in models:
    print(m.name)
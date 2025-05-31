import os
import langchain
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(langchain.__version__)
print("API Key:", GEMINI_API_KEY)  # TEMPORARILY to debug, remove later
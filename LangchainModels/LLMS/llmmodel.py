from langchain_openai import OpenAI
from dotenv import load_dotenv #loading secret keys

load_dotenv() # Load environment variables from .env file

llm =OpenAI(model="gpt-3.5-turbo-instruct")

result= llm.invoke("what is the capital of nepal?")

print(result)

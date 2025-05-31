from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv  # loading secret keys

load_dotenv()  # Load environment variables from .env file

Embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

result = Embedding.embed_query("What is the capital of Nepal?")

print(str(result))  # Output the embedding vector
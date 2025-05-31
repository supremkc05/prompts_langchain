from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "What is the capital of Nepal?"
result = embedding.embed_query(text)
print(str(result))  # Output the embedding vector
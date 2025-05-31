from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm  = HuggingFaceEndpoint(
    repo_id ="mradermacher/TinyLamma1.1b-finetuned-GGUF",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
result =model.invoke("what is the pm of nepal?")
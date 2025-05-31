import streamlit as st
from dotenv import load_dotenv
import os
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Hugging Face model config
llm = HuggingFaceHub(
    repo_id="sshleifer/distilbart-cnn-12-6",
    model_kwargs={"temperature": 0.3, "max_new_tokens": 500},
    huggingfacehub_api_token=hf_token,
)

# Prompt template
template = PromptTemplate(
    input_variables=["paper", "style", "length"],
    template="""
    Summarize the following research paper in a {style} way with a {length} level of detail.

    Research Paper:
    {paper}
    """
)

# LangChain pipeline
chain = LLMChain(llm=llm, prompt=template)

# Streamlit UI
st.title("📝 Free Text Summarizer using Hugging Face")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# Sample paper text – replace with real paper content or PDF reader later
papers = {
    "Attention Is All You Need": "This paper introduces the Transformer model...",
    "BERT: Pre-training of Deep Bidirectional Transformers": "BERT is a method of pretraining language representations...",
    "GPT-3: Language Models are Few-Shot Learners": "GPT-3 is a language model developed by OpenAI with 175 billion parameters...",
    "Diffusion Models Beat GANs on Image Synthesis": "Diffusion models are a class of generative models..."
}

# Button trigger
if st.button("Summarize"):
    paper_text = papers[paper_input]
    output = chain.run({
        "paper": paper_text,
        "style": style_input,
        "length": length_input
    })
    st.subheader("📄 Summary")
    st.write(output)

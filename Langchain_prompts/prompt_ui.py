import os
import streamlit as st
from dotenv import load_dotenv
import os
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables (Hugging Face token)
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Validate token
if not hf_token:
    st.error("⚠️ Hugging Face API token not found. Please set it in your .env file.")
    st.stop()

# UI
st.title("📄 Free Research Paper Summarizer (Hugging Face)")

# Inputs
paper_input = st.selectbox(
    "Choose Research Paper",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Choose Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Choose Summary Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# Dummy content
papers = {
    "Attention Is All You Need": """
        The 'Attention Is All You Need' paper proposes the Transformer model, 
        which does not rely on recurrent layers or convolutions. Instead, it uses self-attention mechanisms 
        to process sequences in parallel, making training faster and more scalable. The paper demonstrates 
        state-of-the-art performance in machine translation tasks and forms the basis for many modern NLP models.
    """,
    "BERT: Pre-training of Deep Bidirectional Transformers": """
        BERT introduces a new language representation model designed to pre-train deep bidirectional representations 
        by jointly conditioning on both left and right context. It achieves state-of-the-art results on a wide array 
        of NLP tasks by fine-tuning with just one additional output layer.
    """,
    "GPT-3: Language Models are Few-Shot Learners": """
        GPT-3, developed by OpenAI, is a massive transformer-based model trained on 175 billion parameters. 
        It is capable of performing NLP tasks with little to no fine-tuning by simply providing instructions or examples 
        in the input prompt, demonstrating strong few-shot learning abilities.
    """,
    "Diffusion Models Beat GANs on Image Synthesis": """
        This paper shows that diffusion models, which generate images by reversing a noise process, outperform GANs 
        in terms of sample quality. The authors introduce improved training techniques and evaluation metrics that make 
        diffusion models a competitive alternative for high-quality image generation.
    """
}

# Prompt
prompt = PromptTemplate(
    input_variables=["paper", "style", "length"],
    template="""
    You are an expert summarizer. Given this research paper content, create a summary in a {style} tone.
    The summary should be {length}.

    Research Paper:
    {paper}
    """
)

# Use Hugging Face LLM (Free)
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",  # ✅ Free and fast model
    model_kwargs={"temperature": 0.3, "max_length": 512},
    huggingfacehub_api_token=hf_token
)

# LangChain chain
chain = LLMChain(prompt=prompt, llm=llm)

# Summarize
if st.button("Summarize"):
    with st.spinner("Generating summary..."):
        paper_text = papers[paper_input]
        summary = chain.run({
            "paper": paper_text,
            "style": style_input,
            "length": length_input
        })
        st.subheader("📝 Summary")
        st.write(summary)

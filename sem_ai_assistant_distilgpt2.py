
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.title("🧠 SEM Ad Copy Generator (Streamlit + distilgpt2)")

@st.cache_resource
def load_model():
    model_name = 'distilgpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.header("Generate Ad Copy")
prompt = st.text_input("Describe your product, audience, and offer")
if prompt:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.text_area("Generated Ad Text", response, height=150)

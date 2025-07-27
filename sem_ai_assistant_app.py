
import streamlit as st
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
@st.cache_resource
def load_model():
    model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("🧠 SEM AI Assistant")

# Ad Generator
st.header("1. Generate Ad Copy")
user_prompt = st.text_input("Describe your product/service and goals")
if user_prompt:
    inputs = tokenizer(user_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=80)
    st.text_area("Generated Ad Copy", tokenizer.decode(outputs[0], skip_special_tokens=True), height=200)

# Campaign Analyzer
st.header("2. Analyze Campaign Performance")
uploaded_file = st.file_uploader("Upload your Google Ads keyword CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", df.head())
    flagged = df[(df['Cost ($)'] > 50) & (df['Conversions'] < 2)]
    st.subheader("Flagged Underperforming Keywords")
    st.write(flagged[['Keyword/Asset', 'Cost ($)', 'Conversions', 'ROAS']])

# Optimizer
st.header("3. Optimization Suggestions")
cost = st.number_input("Enter cost for a keyword/campaign", value=0.0)
conversions = st.number_input("Enter number of conversions", value=0)
ctr = st.number_input("Enter CTR (%)", value=0.0)

if st.button("Suggest Actions"):
    actions = []
    if cost > 100 and conversions < 1:
        actions.append("Pause low-converting keyword.")
    if ctr < 1.5:
        actions.append("Test new headlines focused on urgency or social proof.")
    if actions:
        st.write("Recommended Actions:")
        for act in actions:
            st.markdown(f"- {act}")
    else:
        st.write("No immediate actions needed.")

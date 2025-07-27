
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.header("🧠 Ad Copy Generator")

@st.cache_resource
def load_model():
    model_name = 'distilgpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

prompt = st.text_input("Describe your product, audience, and offer")
if prompt:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.text_area("Generated Ad Text", response, height=150)



import streamlit as st
import pandas as pd

st.header("📊 Campaign Data Analyzer")

uploaded_file = st.file_uploader("Upload a Google Ads CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview", df.head())

    st.subheader("Underperforming Keywords")
    flagged = df[(df['Cost ($)'] > 50) & (df['Conversions'] < 2)]
    st.write(flagged[['Keyword/Asset', 'Cost ($)', 'Conversions', 'ROAS']])



import streamlit as st

st.header("🔧 Optimization Suggestion Tool")

cost = st.number_input("Keyword/Ad Cost ($)", value=0.0)
conversions = st.number_input("Conversions", value=0)
ctr = st.number_input("Click Through Rate (%)", value=0.0)

if st.button("Suggest Actions"):
    suggestions = []
    if cost > 100 and conversions < 1:
        suggestions.append("⚠️ Pause low-converting keyword.")
    if ctr < 1.5:
        suggestions.append("💡 Test new headlines with urgency or proof.")
    if suggestions:
        for s in suggestions:
            st.write(s)
    else:
        st.success("No critical issues detected.")

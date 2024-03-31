# import tracemalloc
import streamlit as st 
from transformers import pipeline

model = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

# tracemalloc.start()

st.set_page_config(
    page_title="Mistral Chat",
    page_icon=":robot:"
)

prompt = st.text_input("Prompt:")  

output = model(prompt)

st.write(output[0]["generated_text"])
# st.write(tracemalloc.get_traced_memory())
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Check if GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

local_model_path = 'models/'

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

# Send the model to the device (GPU if available, otherwise CPU)
model.to(device)

# Streamlit app
def main():
    st.title("Code Completion with bigcode/starcoder")

    # Text input
    input_text = st.text_area("Enter your code snippet here:", height=150)
    submit_button = st.button("Generate Completion")

    if submit_button:
        with st.spinner('Generating...'):
            # Encoding input text
            input_ids = tokenizer.encode(input_text, return_tensors="pt")
            input_ids = input_ids.to(device)  # Send input to the same device as model

            # Generating completion
            output = model.generate(input_ids, max_length=50, num_return_sequences=1)

            # Decoding and displaying the generated text
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            st.text_area("Generated Code:", value=generated_text, height=150, key="output")

main()

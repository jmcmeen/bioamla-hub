# Streamlit Audio Classifier using Audio Spectrogram Transformer
import streamlit as st
from apps.torch_ast import wave_file_ast_inference

# Streamlit configuration
st.set_page_config(page_title="Audio Classifier", layout="wide")

# Set page title
st.title("Audio Classifier using Audio Spectrogram Transformer")

# Write description
st.write("This app uses the following pretrained model: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593")

# Create upload widget for image
uploaded_file = st.file_uploader("Choose a wave file, please!", type=["wav"])
model = "MIT/ast-finetuned-audioset-10-10-0.4593"
resample_freq = 16000

# Make prediction
if uploaded_file is not None:
    prediction = wave_file_ast_inference(uploaded_file, model, resample_freq)
    output = "Prediction: " + prediction
    st.write(output)
import streamlit as st
import torch
import os
import urllib.request
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Page Config
st.set_page_config(page_title="Emotion AI Demo", page_icon="🤖")
st.title("🎭 Multi-Label Emotion Detection")
st.markdown("Developed as part of my **B.Sc. AI & ML Research Project**.")

# 2. Define where to save the model on the Streamlit server
SAVE_DIR = "model_files"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 3. Use your Release links
# Replace these with the actual "Copy Link Address" from your GitHub Release page
MODEL_URL = "https://github.com/heera9721/Emotion-Detection-Project/releases/download/v1.0/model.safetensors"
CONFIG_URL = "https://github.com/heera9721/Emotion-Detection-Project/releases/download/v1.0/config.json"

@st.cache_resource
def load_model():
    model_file = os.path.join(SAVE_DIR, "model.safetensors")
    config_file = os.path.join(SAVE_DIR, "config.json")

    # Download if not present
    if not os.path.exists(model_file):
        with st.spinner("Downloading AI Model from GitHub Releases... This only happens once."):
            urllib.request.urlretrieve(MODEL_URL, model_file)
            urllib.request.urlretrieve(CONFIG_URL, config_file)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    # Load from the local folder we just created
    model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
    model.to("cpu")
    return tokenizer, model

# Trigger loading
try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Model failed to load. Error: {e}")
    st.stop()

# 4. Emotion labels
emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", 
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
    "joy", "love", "nervousness", "optimism", "pride", "realization", 
    "relief", "remorse", "sadness", "surprise", "neutral"
]

user_input = st.text_area("Enter text to analyze:", placeholder="e.g., I'm so proud of this project!")

if st.button("Analyze Emotion"):
    if user_input:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
        
        st.subheader("Results:")
        found = False
        for i, p in enumerate(probs):
            if p > 0.3:
                st.write(f"**{emotions[i].upper()}** ({round(float(p)*100, 2)}%)")
                st.progress(float(p))
                found = True
        
        if not found:
            st.info("The AI detected this as **NEUTRAL**.")
    else:
        st.warning("Please enter some text first!")

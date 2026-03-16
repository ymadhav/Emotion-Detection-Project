import streamlit as st
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up page config
st.set_page_config(page_title="Emotion AI Demo", page_icon="🤖")

st.title("🎭 Multi-Label Emotion Detection")
st.markdown("Developed as part of my **B.Sc. AI & ML Research Project**. This AI can detect 28 different emotions in text.")

# Path to your model

model_path = os.path.join(os.path.dirname(__file__), "EmotionDetection", "real_model")

@st.cache_resource # This keeps the model in memory so it doesn't reload every time
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# Emotion labels
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
        
        probs = torch.sigmoid(outputs.logits).squeeze().numpy()
        
        # Display results
        st.subheader("Results:")
        found = False
        for i, p in enumerate(probs):
            if p > 0.3:
                # Show a progress bar for each detected emotion
                st.write(f"**{emotions[i].upper()}** ({round(p*100, 2)}%)")
                st.progress(float(p))
                found = True
        
        if not found:
            st.info("The AI detected this as **NEUTRAL**.")
    else:
        st.warning("Please enter some text first!")
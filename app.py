import streamlit as st
import torch
import os
import urllib.request
import pandas as pd
import altair as alt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Page Configuration (Wide layout for better visualization)
st.set_page_config(page_title="Emotion AI Research Lab", page_icon="🎭", layout="wide")

# Custom CSS to make it look professional
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_stdio=True)

st.title("🎭 Multi-Label Emotion Detection Dashboard")
st.markdown("### Research Project: Emotion Detection from Text Using NLP and Deep Learning")

# 2. Setup Local Storage & Download Logic
SAVE_DIR = "model_files"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Using your specific GitHub Release links
MODEL_URL = "https://github.com/ymadhav/Emotion-Detection-Project/releases/download/v1.0/model.safetensors"
CONFIG_URL = "https://github.com/ymadhav/Emotion-Detection-Project/releases/download/v1.0/config.json"

@st.cache_resource
def load_model():
    model_file = os.path.join(SAVE_DIR, "model.safetensors")
    config_file = os.path.join(SAVE_DIR, "config.json")

    if not os.path.exists(model_file):
        with st.spinner("Downloading AI Model from GitHub Releases... This only happens once."):
            urllib.request.urlretrieve(MODEL_URL, model_file)
            urllib.request.urlretrieve(CONFIG_URL, config_file)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
    model.to("cpu")
    return tokenizer, model

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Model failed to load. Ensure the GitHub Repository is Public. Error: {e}")
    st.stop()

# 3. Emotion Metadata & Sentiment Mapping
emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", 
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
    "joy", "love", "nervousness", "optimism", "pride", "realization", 
    "relief", "remorse", "sadness", "surprise", "neutral"
]

sentiment_groups = {
    "Positive": ["admiration", "amusement", "approval", "caring", "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief"],
    "Negative": ["anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"],
    "Ambiguous": ["confusion", "curiosity", "desire", "realization", "surprise"]
}

# 4. Sidebar Information
with st.sidebar:
    st.header("Project Overview")
    st.info("This AI utilizes a fine-tuned **DistilBERT** model to perform multi-label classification across 28 emotional states.")
    st.write("**Dataset:** GoEmotions")
    st.write("**Frameworks:** Transformers, PyTorch, Streamlit")
    st.divider()
    st.write("Developed by: [Your Name]")

# 5. Main Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Input Text Analysis")
    user_input = st.text_area("Enter text to analyze emotional depth:", height=150, placeholder="e.g., I am so grateful for the opportunity to work on this research project!")
    analyze_btn = st.button("🚀 Run AI Inference")

if analyze_btn and user_input:
    # Inference
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

    # Process Results
    results_df = pd.DataFrame({"Emotion": emotions, "Score": probs})
    top_emotions = results_df[results_df["Score"] > 0.3].sort_values(by="Score", ascending=False)

    with col2:
        st.subheader("Results & Visualization")
        if not top_emotions.empty:
            # Display Primary Metric
            primary = top_emotions.iloc[0]["Emotion"].upper()
            st.metric(label="Primary Emotion Detected", value=primary)

            # Altair Chart for Research Presentation
            chart = alt.Chart(top_emotions).mark_bar().encode(
                x=alt.X('Score:Q', title='Confidence Score'),
                y=alt.Y('Emotion:N', sort='-x', title='Emotions'),
                color=alt.Color('Score:Q', scale=alt.Scale(scheme='viridis'))
            ).properties(height=300)
            
            st.altair_chart(chart, use_container_width=True)

            # Sentiment Categorization
            detected = top_emotions["Emotion"].tolist()
            pos_found = [e for e in detected if e in sentiment_groups["Positive"]]
            neg_found = [e for e in detected if e in sentiment_groups["Negative"]]
            
            if pos_found: st.success(f"**Positive Indicators:** {', '.join(pos_found)}")
            if neg_found: st.error(f"**Negative Indicators:** {', '.join(neg_found)}")
        else:
            st.info("The AI detected this as **NEUTRAL** (no strong emotions above 30% threshold).")

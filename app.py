import streamlit as st
import torch
import os
import urllib.request
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Page Configuration
st.set_page_config(page_title="Emotion AI Research Lab", page_icon="🎭", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    [data-testid="stMetricValue"] { color: #1f77b4 !important; }
    [data-testid="stMetricLabel"] { color: #31333F !important; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.05); border: 1px solid #e6e9ef; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎭 Multi-Label Emotion Detection Dashboard")
st.markdown("### Research Project: Emotion Detection from Text Using NLP and Deep Learning")

# 2. Setup Local Storage & Download Logic
SAVE_DIR = "model_files"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

MODEL_URL = "https://github.com/ymadhav/Emotion-Detection-Project/releases/download/v1.0/model.safetensors"
CONFIG_URL = "https://github.com/ymadhav/Emotion-Detection-Project/releases/download/v1.0/config.json"

@st.cache_resource
def load_model():
    model_file = os.path.join(SAVE_DIR, "model.safetensors")
    config_file = os.path.join(SAVE_DIR, "config.json")

    if not os.path.exists(model_file):
        with st.spinner("Downloading AI Model from GitHub Releases..."):
            urllib.request.urlretrieve(MODEL_URL, model_file)
            urllib.request.urlretrieve(CONFIG_URL, config_file)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
    model.to("cpu")
    return tokenizer, model

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Model failed to load. Error: {e}")
    st.stop()

# 3. Emotion Metadata
emotions = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity","desire",
    "disappointment","disapproval","disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"
]

sentiment_groups = {
    "Positive": ["admiration","amusement","approval","caring","excitement","gratitude","joy","love","optimism","pride","relief"],
    "Negative": ["anger","annoyance","disappointment","disapproval","disgust","embarrassment","fear","grief","nervousness","remorse","sadness"],
    "Ambiguous": ["confusion","curiosity","desire","realization","surprise"]
}

# 4. Sidebar
with st.sidebar:
    st.header("Project Overview")
    st.info("Fine-tuned DistilBERT for multi-label classification across 28 emotions.")
    st.write("**Dataset:** GoEmotions")
    st.write("**Frameworks:** Transformers, PyTorch, Streamlit")
    st.divider()
    mode = st.radio("Choose Mode:", ["Inference", "Evaluation"])

# 5. Inference Mode
if mode == "Inference":
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Input Text Analysis")
        user_input = st.text_area("Enter text:", height=150, placeholder="e.g., I am so grateful for this opportunity!")
        analyze_btn = st.button("🚀 Run AI Inference")

    if analyze_btn and user_input:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

        results_df = pd.DataFrame({"Emotion": emotions, "Score": probs})
        top_emotions = results_df[results_df["Score"] > 0.3].sort_values(by="Score", ascending=False)

        with col2:
            st.subheader("Results & Visualization")
            if not top_emotions.empty:
                primary = top_emotions.iloc[0]["Emotion"].upper()
                st.metric(label="Primary Emotion Detected", value=primary)

                chart = alt.Chart(top_emotions).mark_bar().encode(
                    x=alt.X('Score:Q', title='Confidence Score'),
                    y=alt.Y('Emotion:N', sort='-x', title='Emotions'),
                    color=alt.Color('Score:Q', scale=alt.Scale(scheme='viridis'))
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

                detected = top_emotions["Emotion"].tolist()
                pos_found = [e for e in detected if e in sentiment_groups["Positive"]]
                neg_found = [e for e in detected if e in sentiment_groups["Negative"]]

                if pos_found: st.success(f"**Positive Indicators:** {', '.join(pos_found)}")
                if neg_found: st.error(f"**Negative Indicators:** {', '.join(neg_found)}")
            else:
                st.info("Detected as **NEUTRAL** (no strong emotions above threshold).")

# 6. Evaluation Mode
elif mode == "Evaluation":
    st.subheader("📊 Dataset Emotion Distribution")
    try:
        df = pd.read_csv("goemtion1.csv")
        emotion_counts = df['labels'].value_counts()
        st.bar_chart(emotion_counts)
    except Exception as e:
        st.warning(f"Dataset not found: {e}")

    st.subheader("📈 Model Performance Metrics")
    # Example dummy values — replace with actual test set labels/preds
    y_true = ["joy","sadness","anger","joy","fear"]
    y_pred = ["joy","sadness","anger","neutral","fear"]

    report = classification_report(y_true, y_pred, target_names=list(set(y_true)), output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.subheader("🔍 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred, labels=list(set(y_true)))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=list(set(y_true)), yticklabels=list(set(y_true)))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig)

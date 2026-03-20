import streamlit as st
import torch
import os
import urllib.request
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
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

# 6. Evaluation Mode (Multi-label)
elif mode == "Evaluation":
    st.subheader("📊 Dataset Emotion Distribution")
    try:
        # Directly load datasets from GitHub release URLs
        url1 = "https://github.com/ymadhav/Emotion-Detection-Project/releases/download/v1.0/goemotions_1.csv"
        url2 = "https://github.com/ymadhav/Emotion-Detection-Project/releases/download/v1.0/goemotions_2.csv"
        url3 = "https://github.com/ymadhav/Emotion-Detection-Project/releases/download/v1.0/goemotions_3.csv"

        df1 = pd.read_csv(url1)
        df2 = pd.read_csv(url2)
        df3 = pd.read_csv(url3)
        test_df = pd.concat([df1, df2, df3], ignore_index=True)

        emotion_counts = test_df['labels'].value_counts()
        st.bar_chart(emotion_counts)

        # Prepare ground truth (assuming labels column has comma-separated emotions)
        y_true = [lbl.split(",") for lbl in test_df["labels"].tolist()]

        y_pred = []
        threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.3)

        for text in test_df["text"]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

            pred_emotions = [emotions[i] for i, p in enumerate(probs) if p > threshold]
            y_pred.append(pred_emotions)

        # Convert to binary indicator arrays
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

        mlb = MultiLabelBinarizer(classes=emotions)
        y_true_bin = mlb.fit_transform(y_true)
        y_pred_bin = mlb.transform(y_pred)

        # Compute micro-averaged metrics
        precision = precision_score(y_true_bin, y_pred_bin, average="micro")
        recall = recall_score(y_true_bin, y_pred_bin, average="micro")
        f1 = f1_score(y_true_bin, y_pred_bin, average="micro")

        st.subheader("📈 Model Performance Metrics")
        st.metric("Precision", f"{precision:.3f}")
        st.metric("Recall", f"{recall:.3f}")
        st.metric("F1 Score", f"{f1:.3f}")

        # Per-class breakdown
        report = classification_report(y_true_bin, y_pred_bin, target_names=emotions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # Per-class F1 bar chart
        f1_scores = [report[e]["f1-score"] for e in emotions if e in report]
        chart_df = pd.DataFrame({"Emotion": emotions, "F1": f1_scores})
        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("F1:Q", title="F1 Score"),
            y=alt.Y("Emotion:N", sort="-x"),
            color=alt.Color("F1:Q", scale=alt.Scale(scheme="viridis"))
        ).properties(height=400)
        st.subheader("📊 Per-Class F1 Scores")
        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.warning(f"Evaluation failed: {e}")

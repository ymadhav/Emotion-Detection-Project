
# 🎭 Multi-Label Emotion Detection Dashboard

![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-orange?style=flat-square) ![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square) ![Powered by](https://img.shields.io/badge/Powered_by-DistilBERT-red?style=flat-square) ![UI](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square)

The Multi-Label Emotion Detection Dashboard is an advanced Natural Language Processing system built for high-accuracy text analysis. It uses a fine-tuned **DistilBERT** transformer model to categorize, analyze, and visualize text into 28 distinct emotional states, ensuring a nuanced understanding of human communication through a custom tokenization architecture.

---

## 🌟 Key Innovation: Sentence-Level Max-Pooling

Unlike standard document-level AI classifiers, this project includes an **Aggregation Layer** that acts as a sentiment preserver:

* **Dynamic Tokenization:** Automatically splits complex paragraphs into individual sentences *before* they reach the transformer.
* **Independent Evaluation:** Assigns a mathematical confidence score across all 28 emotional dimensions for every single sentence.
* **Peak Aggregation:** Employs Max-Pooling to capture the strongest emotions, completely bypassing the "Sentiment Dilution" effect found in traditional NLP models where conflicting emotions cancel each other out.

---

## 🚀 Live Demo

Try the live application here: [Emotion Detection on Streamlit](https://emotion-detection-project-3svjxkorm8anv5mktkp7ae.streamlit.app)

*(Note: The live demo is powered by a PyTorch backend, executing sub-second inference per input block).*

---

## 🏗️ System Architecture & Features

This project moves beyond standard ML implementations by building a robust, research-grade pipeline:

1. **Fine-Tuned Transformer Backbone:** Utilizes a DistilBERT model trained on the GoEmotions dataset, achieving a proven 91.05% validation accuracy.
2. **Threshold Filtering:** Applies a 0.3 Sigmoid function threshold to successfully identify and surface multiple co-occurring emotions simultaneously.
3. **Cloud-Native Weight Management:** Bypasses standard Git LFS and repository storage constraints by dynamically fetching model weights (260MB+) from the GitHub Releases API upon initialization.
4. **Declarative Visualizations:** Leverages the Altair library to render real-time, interactive confidence scores and primary emotion metrics directly in the UI.

```

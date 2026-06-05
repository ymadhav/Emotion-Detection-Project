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

---

## 📊 Model Performance

The network converged effectively over 10 training epochs with an empirical classification cutoff threshold set at `0.3` to optimize the precision-recall balance.

| Metric | Training Phase (Epoch 10) | Validation Phase (Epoch 10) |
| :--- | :--- | :--- |
| **Loss** | 0.2241 | 0.2315 |
| **Accuracy** | **92.45%** | **91.05%** |
| **Precision** | 0.92 | 0.91 |
| **Recall** | 0.93 | 0.91 |
| **F1-Score** | 0.92 | 0.91 |

---

## 💻 Local Setup & Installation (Beginner-Friendly)

Follow these step-by-step instructions to run the project on your own computer.

### Step 1: Install Prerequisites
* Download and install **Python 3.10** (or newer) from the official Python website.
* Download and install **Git** to manage the repository files.
* Download and install a code editor like **VS Code**.

### Step 2: Clone the Repository
Open your computer's terminal (or command prompt) and download the project files:
```bash
git clone [https://github.com/ymadhav/Emotion-Detection-Project.git](https://github.com/ymadhav/Emotion-Detection-Project.git)

```

Navigate into the newly created project folder:

```bash
cd Emotion-Detection-Project

```

### Step 3: Create a Virtual Environment

Isolate the project dependencies from your main system.

* **On Windows:**

```bash
python -m venv venv

```

* **On macOS / Linux:**

```bash
python3 -m venv venv

```

### Step 4: Activate the Virtual Environment

Turn on the environment before installing any packages.

* **On Windows:**

```bash
venv\Scripts\activate

```

* **On macOS / Linux:**

```bash
source venv/bin/activate

```

### Step 5: Install Required Libraries

Install all the necessary Python packages (such as Streamlit, PyTorch, and Transformers) listed in your configuration file:

```bash
pip install -r requirements.txt

```

### Step 6: Run the Application

Start the local server to launch the web dashboard:

```bash
streamlit run app.py

```

Open the local URL provided in your terminal (usually `http://localhost:8501`) in your web browser.

*(Note: The application will automatically download the required model weights from GitHub Releases upon the first launch, which may take a few moments depending on your internet speed).*

---

## 🤝 Contributing

Contributions, code enhancements, and issue reports are welcome.

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a formal Pull Request.

```

```

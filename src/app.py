import streamlit as st
import requests
import os
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# === CONFIGURATION ===
MODEL_DIR = "bert_sentiment_model"
BASE_URL = "https://github.com/kaushalnandaniya/sentiment_analysis/releases/download/bert"
MODEL_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt"
]

# === DOWNLOAD MODEL FILES FROM GITHUB RELEASE ===
def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for file in MODEL_FILES:
        file_path = os.path.join(MODEL_DIR, file)
        if not os.path.exists(file_path):
            st.write(f"📥 Downloading `{file}`...")
            url = f"{BASE_URL}/{file}"
            r = requests.get(url)
            if r.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(r.content)
            else:
                st.error(f"❌ Failed to download {file} from {url}")
                raise Exception(f"Download failed: {url}")

# === LOAD MODEL & TOKENIZER ===
@st.cache_resource
def load_model():
    download_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# === INFERENCE FUNCTION ===
def predict_sentiment(text, classifier):
    result = classifier(text)[0]
    return result['label'], result['score']

# === STREAMLIT UI ===
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide"
)

st.title("🐦 Twitter Sentiment Analysis")
st.markdown("Analyze the sentiment of tweets and text using AI!")

# Load model
with st.spinner("Loading AI model..."):
    classifier = load_model()

# Sidebar Info
st.sidebar.header("📊 About")
st.sidebar.markdown("""
This app uses a **custom fine-tuned BERT model** hosted on GitHub to analyze tweet-like text.

**Features:**
- 🎯 Accurate predictions on social text
- ⚡ Fast inference
- 🧠 Powered by Transformers
""")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Your Text")

    example_tweets = [
        "I love this new phone! The camera is amazing! 📱",
        "This restaurant was terrible. Worst food ever! 😤",
        "Just had the best coffee of my life! ☕",
        "The movie was okay, nothing special.",
        "Customer service was absolutely horrible! 😡"
    ]

    user_input = st.text_area(
        "Enter your tweet or text here:",
        value=example_tweets[0],
        height=120,
        placeholder="Type your text here..."
    )

    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                time.sleep(0.5)
                sentiment, confidence = predict_sentiment(user_input, classifier)
                st.success("Analysis Complete!")

                if sentiment.upper() == "POSITIVE":
                    st.markdown(f"### 😊 **Positive** ({confidence:.1%})")
                    st.progress(confidence)
                elif sentiment.upper() == "NEGATIVE":
                    st.markdown(f"### 😞 **Negative** ({confidence:.1%})")
                    st.progress(confidence)
                else:
                    st.markdown(f"### 😐 **Neutral** ({confidence:.1%})")
                    st.progress(confidence)

                if confidence > 0.8:
                    st.info("🎯 High confidence prediction")
                elif confidence > 0.6:
                    st.warning("⚠️ Medium confidence prediction")
                else:
                    st.warning("🤔 Low confidence prediction")
        else:
            st.error("Please enter some text to analyze!")

with col2:
    st.subheader("🎯 Try Examples")
    for i, tweet in enumerate(example_tweets):
        if st.button(f"Example {i+1}", key=f"example_{i}"):
            st.session_state.example_text = tweet
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit and Hugging Face Transformers</p>
    <p>Model loaded from GitHub Releases</p>
</div>
""", unsafe_allow_html=True)

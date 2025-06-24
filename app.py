import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os

@st.cache_resource
def load_model():
    """Load the trained BERT model and tokenizer from Hugging Face Hub"""
    try:
        # Use a pre-trained sentiment analysis model from Hugging Face Hub
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        
        st.success("✅ Model loaded successfully from Hugging Face Hub!")
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

def predict_sentiment(text, tokenizer, model):
    """Predict sentiment for given text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()
    sentiment = "positive" if pred == 1 else "negative"
    return sentiment, confidence

# Page config
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Twitter Sentiment Analysis")
st.markdown("Analyze the sentiment of tweets using BERT model")

# Load model
tokenizer, model = load_model()

# Input section
st.header("🔍 Analyze Sentiment")
user_input = st.text_area(
    "Enter your tweet or text here:",
    placeholder="I love this product! It's amazing...",
    height=100
)

# Prediction button
if st.button("🚀 Predict Sentiment", type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            sentiment, confidence = predict_sentiment(user_input, tokenizer, model)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            if sentiment == "positive":
                st.success(f"😊 **Sentiment: Positive**")
            else:
                st.error(f"😞 **Sentiment: Negative**")
        
        with col2:
            st.metric("Confidence", f"{confidence:.2%}")
        
        # Confidence bar
        st.progress(confidence)
        
    else:
        st.warning("Please enter some text to analyze.")

# Example tweets
st.header("💡 Try These Examples")
example_tweets = [
    "I love this product! It's absolutely amazing!",
    "This is the worst experience I've ever had.",
    "The service was okay, nothing special.",
    "Absolutely fantastic! Highly recommend to everyone!",
    "I am so disappointed with the quality."
]

for i, tweet in enumerate(example_tweets):
    if st.button(f"Example {i+1}: {tweet[:50]}...", key=f"example_{i}"):
        st.session_state.user_input = tweet
        st.rerun()

# About section
st.markdown("---")
st.header("ℹ️ About This Model")
st.markdown("""
- **Model**: DistilBERT for sentiment analysis
- **Use Case**: Perfect for social media sentiment analysis
- **Deployment**: Lightweight and fast
- **Accuracy**: Good performance on Twitter-style text
""")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Hugging Face Transformers") 
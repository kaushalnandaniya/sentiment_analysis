import streamlit as st
from transformers import pipeline
import time

# Load the sentiment analysis pipeline
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def predict_sentiment(text, classifier):
    result = classifier(text)[0]
    return result['label'], result['score']

# Page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide"
)

# Main app
st.title("🐦 Twitter Sentiment Analysis")
st.markdown("Analyze the sentiment of tweets and text using AI!")

# Load model
with st.spinner("Loading AI model..."):
    classifier = load_model()

# Sidebar
st.sidebar.header("📊 About")
st.sidebar.markdown("""
This app uses a **Twitter-RoBERTa** model fine-tuned for sentiment analysis on social media text.

**Features:**
- 🎯 High accuracy on Twitter-style text
- ⚡ Real-time predictions
- 📱 Mobile-friendly interface
- 🎨 Beautiful UI with emojis
""")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Your Text")
    
    # Example tweets
    example_tweets = [
        "I love this new phone! The camera is amazing! 📱",
        "This restaurant was terrible. Worst food ever! 😤",
        "Just had the best coffee of my life! ☕",
        "The movie was okay, nothing special.",
        "Customer service was absolutely horrible! 😡"
    ]
    
    # Text input
    user_input = st.text_area(
        "Enter your tweet or text here:",
        value=example_tweets[0],
        height=120,
        placeholder="Type your text here..."
    )
    
    # Predict button
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                # Add a small delay for better UX
                time.sleep(0.5)
                
                sentiment, confidence = predict_sentiment(user_input, classifier)
                
                # Display results
                st.success("Analysis Complete!")
                
                # Color-coded sentiment display
                if sentiment == "POSITIVE":
                    st.markdown(f"### 😊 **Positive** ({confidence:.1%})")
                    st.progress(confidence)
                elif sentiment == "NEGATIVE":
                    st.markdown(f"### 😞 **Negative** ({confidence:.1%})")
                    st.progress(confidence)
                else:
                    st.markdown(f"### 😐 **Neutral** ({confidence:.1%})")
                    st.progress(confidence)
                
                # Confidence explanation
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
    st.markdown("Click any example to test the model:")
    
    for i, tweet in enumerate(example_tweets):
        if st.button(f"Example {i+1}", key=f"example_{i}"):
            st.session_state.example_text = tweet
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit and Hugging Face Transformers</p>
    <p>Model: <code>cardiffnlp/twitter-roberta-base-sentiment-latest</code></p>
</div>
""", unsafe_allow_html=True) 

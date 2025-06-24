import os
import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

@st.cache_resource
def load_model(model_dir):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()
    sentiment = "positive" if pred == 1 else "negative"
    return sentiment, confidence

st.title("Twitter Sentiment Analysis (BERT)")
st.write("Enter a tweet below to predict its sentiment using your fine-tuned BERT model.")

model_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'bert_sentiment_model')
tokenizer, model = load_model(model_dir)

user_input = st.text_area("Tweet text", "I love this product! It's amazing.")

if st.button("Predict Sentiment"):
    sentiment, confidence = predict_sentiment(user_input, tokenizer, model)
    st.markdown(f"**Predicted sentiment:** {sentiment}")
    st.markdown(f"**Confidence:** {confidence:.2f}")

st.markdown("---")
st.markdown(
    """
**How to run this app:**

```bash
cd src
streamlit run app.py
```
"""
) 
import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

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

if __name__ == "__main__":
    # Path to your saved model directory
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'bert_sentiment_model')
    tokenizer, model = load_model(model_dir)

    # Example usage
    example_tweets = [
        "I love this product! It's amazing.",
        "This is the worst experience I've ever had.",
        "Not bad, could be better.",
        "Absolutely fantastic! Highly recommend.",
        "I am so disappointed."
    ]
    for tweet in example_tweets:
        sentiment, confidence = predict_sentiment(tweet, tokenizer, model)
        print(f"Tweet: {tweet}\nPredicted sentiment: {sentiment} (confidence: {confidence:.2f})\n") 
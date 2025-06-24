import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

def clean_tweet(text):
    """
    Clean tweet text by removing URLs, mentions, hashtags, special characters, and lowercasing.
    """
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)   # Remove mentions
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)   # Remove hashtags
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = text.lower()                          # Lowercase
    text = re.sub(r'\s+', ' ', text).strip()    # Remove extra spaces
    return text

def tokenize_tweet(text):
    """
    Tokenize tweet text into words.
    """
    return word_tokenize(text)

def preprocess_tweet(text):
    """
    Clean and tokenize a tweet.
    """
    cleaned = clean_tweet(text)
    tokens = tokenize_tweet(cleaned)
    return tokens

if __name__ == "__main__":
    sample = "@user I love #Python! Check out https://python.org :)"
    print("Original:", sample)
    print("Cleaned:", clean_tweet(sample))
    print("Tokens:", preprocess_tweet(sample)) 
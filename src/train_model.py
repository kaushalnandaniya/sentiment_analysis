import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def main():
    # Load cleaned data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_sentiment140.csv')
    df = pd.read_csv(data_path)

    # Prepare features and labels
    X = df['clean_text'].astype(str)
    y = df['target'].map({0: 0, 4: 1})  # 0: negative, 1: positive

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train classifier
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_vec)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred, target_names=['negative', 'positive']))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump({'model': clf, 'vectorizer': vectorizer}, os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiment_model.pkl'))
    print('Model and vectorizer saved to data/sentiment_model.pkl')

if __name__ == "__main__":
    main() 
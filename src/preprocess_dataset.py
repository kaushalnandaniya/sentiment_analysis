import os
import pandas as pd
from tqdm import tqdm
from data_loader import load_sentiment140
from preprocessing import clean_tweet

def preprocess_dataset(input_csv=None, output_csv=None):
    if input_csv is None:
        input_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'training.1600000.processed.noemoticon.csv')
    if output_csv is None:
        output_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_sentiment140.csv')

    print(f"Loading dataset from {input_csv}...")
    df = load_sentiment140(input_csv)
    tqdm.pandas(desc="Cleaning tweets")
    df['clean_text'] = df['text'].progress_apply(clean_tweet)
    print(f"Saving cleaned dataset to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print("Done.")

if __name__ == "__main__":
    preprocess_dataset() 
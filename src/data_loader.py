import pandas as pd
import os

def load_sentiment140(csv_path=None):
    """
    Load the Sentiment140 dataset from a CSV file.
    Args:
        csv_path (str): Path to the CSV file. If None, defaults to 'data/training.1600000.processed.noemoticon.csv'.
    Returns:
        pd.DataFrame: Loaded DataFrame with columns [target, ids, date, flag, user, text]
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training.1600000.processed.noemoticon.csv')
    df = pd.read_csv(csv_path, encoding='latin-1', header=None)
    df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    return df

if __name__ == "__main__":
    df = load_sentiment140()
    print(df.head()) 
import sys
print("Python executable:", sys.executable)
import transformers
print("Transformers version:", transformers.__version__)
print("Transformers loaded from:", transformers.__file__)
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_sentiment140.csv')
df = pd.read_csv(data_path)

# Use a 20,000-sample subset for local training
if len(df) > 20000:
    df = df.sample(20000, random_state=42)

# 2. Prepare labels (0: negative, 1: positive)
df = df[df['target'].isin([0, 4])]
df['label'] = df['target'].map({0: 0, 4: 1})

# 3. Train/test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['clean_text'].astype(str).tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# 4. Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=64)

# 5. Dataset class
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# 6. Model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# 7. Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    save_total_limit=1,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    report = classification_report(labels, preds, target_names=['negative', 'positive'], output_dict=True)
    return {
        'accuracy': report['accuracy'],
        'f1_negative': report['negative']['f1-score'],
        'f1_positive': report['positive']['f1-score'],
    }

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 9. Train
trainer.train()

# 10. Save model and tokenizer
model_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'bert_sentiment_model')
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f'Model and tokenizer saved to {model_dir}') 
# Sentimental: Twitter Sentiment Analysis

A powerful sentiment analysis tool using fine-tuned BERT model trained on the Sentiment140 dataset.

## 🚀 Live Demo

[Deploy your own version](#deployment)

## 📊 Features

- **BERT-based Model**: Fine-tuned DistilBERT for optimal performance
- **High Accuracy**: Trained on 1.6M+ tweets
- **Real-time Analysis**: Instant sentiment predictions
- **Confidence Scores**: See how confident the model is
- **Beautiful UI**: Modern Streamlit interface

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd sentimental
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run locally:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
sentimental/
├── app.py                 # Main Streamlit application
├── src/                   # Source code
│   ├── data_loader.py     # Data loading utilities
│   ├── preprocessing.py   # Text preprocessing
│   ├── train_bert.py      # BERT training script
│   ├── predict_bert.py    # Prediction utilities
│   └── app.py            # Local Streamlit app
├── data/                  # Data and models
│   ├── bert_sentiment_model/  # Trained BERT model
│   └── cleaned_sentiment140.csv
├── notebooks/             # Jupyter notebooks
└── requirements.txt       # Python dependencies
```

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. **Push to GitHub**:
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io]https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file path to `app.py`
   - Click "Deploy"

### Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Upload your code and model
3. Deploy automatically

### Local Network

Run on your local network for team access:
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## 📈 Model Performance

- **Dataset**: Sentiment140 (1.6M tweets)
- **Architecture**: DistilBERT
- **Accuracy**: ~80% on validation set
- **Training Time**: ~30 minutes on GPU

## 🔧 Customization

### Train Your Own Model

1. Prepare your dataset in the same format as Sentiment140
2. Run the training script:
```bash
python src/train_bert.py
```

### Modify the App

Edit `app.py` to:
- Change the UI design
- Add new features
- Integrate with other services

## 📝 License

MIT License - feel free to use this project for your own applications!

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

Built with ❤️ using Streamlit and Hugging Face Transformers 
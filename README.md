# Sentimental: Twitter Sentiment Analysis

A lightweight sentiment analysis tool using advanced AI models, deployed on Streamlit Cloud.


## 📊 Features

- **AI-Powered Analysis**: Uses Twitter-RoBERTa model for accurate sentiment detection
- **Real-time Processing**: Instant sentiment predictions with confidence scores
- **Example Tweets**: Try pre-loaded examples to test the model

## 🛠️ Quick Start

### Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/kaushalnandaniya/sentiment_analysis.git
cd sentiment_analysis
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the app:**
```bash
streamlit run src/app.py
```

4. **Open your browser** and go to `http://localhost:8501`

## 📁 Project Structure

```
sentiment_analysis/
├── src/
│   └── app.py            # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore rules
└── LICENSE               # Apache 2.0 license
```


## 🔧 How It Works

1. **Text Input**: Users enter tweets or text in the text area
2. **AI Processing**: The Twitter-RoBERTa model analyzes the sentiment
3. **Results Display**: Shows sentiment (positive/negative/neutral) with confidence

## 📈 Model Details

- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Training**: Fine-tuned on Twitter data
- **Accuracy**: High performance on social media text
- **Languages**: English (optimized for Twitter-style text)

## 🎯 Use Cases

- **Social Media Monitoring**: Analyze customer sentiment on Twitter
- **Product Reviews**: Understand customer feedback
- **Market Research**: Track brand sentiment over time
- **Content Analysis**: Evaluate the tone of written content


```

## 📝 License

Apache 2.0 License - feel free to use this project for your own applications!




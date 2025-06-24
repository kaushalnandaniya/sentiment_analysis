# Sentimental: Twitter Sentiment Analysis

A lightweight sentiment analysis tool using advanced AI models, deployed on Streamlit Cloud.

## 🚀 Live Demo

[View the live app](https://sentimentanalysis0kaushal.streamlit.app)

## 📊 Features

- **AI-Powered Analysis**: Uses Twitter-RoBERTa model for accurate sentiment detection
- **Real-time Processing**: Instant sentiment predictions with confidence scores
- **Beautiful UI**: Modern Streamlit interface with emojis and progress bars
- **Example Tweets**: Try pre-loaded examples to test the model
- **Mobile Friendly**: Works great on all devices

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

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. **Fork this repository** or create your own
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub repository**
4. **Set main file path to `src/app.py`**
5. **Click "Deploy"**

### Alternative Platforms

- **Hugging Face Spaces**: Upload to HF Spaces for free hosting
- **Heroku**: Use the Procfile for Heroku deployment
- **Railway**: Connect GitHub repo for automatic deployment

## 🔧 How It Works

1. **Text Input**: Users enter tweets or text in the text area
2. **AI Processing**: The Twitter-RoBERTa model analyzes the sentiment
3. **Results Display**: Shows sentiment (positive/negative/neutral) with confidence
4. **Visual Feedback**: Progress bars and emojis for better UX

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

## 🛠️ Customization

### Modify the App

Edit `src/app.py` to:
- Change the UI design
- Add new features (batch processing, file upload)
- Integrate with databases or APIs
- Add more example tweets

### Use Different Models

Replace the model in `src/app.py`:
```python
# For different sentiment models
classifier = pipeline("sentiment-analysis", model="your-model-name")
```

## 📝 License

Apache 2.0 License - feel free to use this project for your own applications!

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

If you encounter any issues:
1. Check the [Streamlit Cloud logs](https://share.streamlit.io)
2. Verify your `requirements.txt` is up to date
3. Test locally before deploying

---

Built with ❤️ using Streamlit and Hugging Face Transformers 

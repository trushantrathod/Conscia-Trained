import nltk
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

# Download VADER
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except:
    nltk.download('vader_lexicon')

# Initialize models (load once)
sia = SentimentIntensityAnalyzer()
bert = pipeline("sentiment-analysis")

# -----------------------------
# VADER SCORE
# -----------------------------
def vader_score(text):
    score = sia.polarity_scores(text)['compound']  # -1 to +1
    return (score + 1) * 50  # → 0 to 100

# -----------------------------
# BERT SCORE
# -----------------------------
def bert_score(text):
    try:
        result = bert(text[:512])[0]

        if result['label'] == "POSITIVE":
            return 50 + result['score'] * 50
        else:
            return 50 - result['score'] * 50
    except:
        return 50

# -----------------------------
# HYBRID SCORE
# -----------------------------
def get_review_score(text):
    v = vader_score(text)
    b = bert_score(text)

    return round((0.4 * v) + (0.6 * b), 2)

# -----------------------------
# FINAL SENTIMENT (REAL-TIME)
# -----------------------------
def get_final_sentiment(base_score, user_review=None):
    if user_review:
        review_score = get_review_score(user_review)

        # 🔥 Combine product reputation + user review
        final_score = (0.7 * base_score) + (0.3 * review_score)
    else:
        final_score = base_score

    return round(max(0, min(100, final_score)), 2)
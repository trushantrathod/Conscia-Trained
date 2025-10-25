import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

# --- Download VADER Lexicon (if needed on a new machine) ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER lexicon for sentiment analysis (one-time setup)...")
    nltk.download('vader_lexicon')

def get_vader_classification(reviews_text):
    """
    Fallback: Classifies text as positive, negative, or neutral using VADER.
    """
    if not reviews_text or pd.isna(reviews_text):
        return "neutral"

    sid = SentimentIntensityAnalyzer()
    compound_score = sid.polarity_scores(str(reviews_text))['compound']
    
    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    else:
        return "neutral"


def get_sentiment_classification(reviews_text, gemini_model):
    """
    Calculates a sentiment classification using Gemini if available,
    otherwise falls back to VADER.
    """
    if not reviews_text or pd.isna(reviews_text):
        return "neutral"

    # --- Fallback 1: If Gemini API key is missing ---
    if not gemini_model:
        return get_vader_classification(reviews_text)

    # --- New Gemini Logic ---
    try:
        # NEW PROMPT: Ask for a classification, not a score.
        prompt = f"""
        You are a sentiment analysis expert. Analyze the following product review.
        Is the review's sentiment positive, negative, or neutral?

        Review:
        "{reviews_text}"

        Respond ONLY with one single word: positive, negative, or neutral.
        """
        
        response = gemini_model.generate_content(prompt)
        
        # Clean up the response
        classification = response.text.strip().lower()

        if "positive" in classification:
            return "positive"
        elif "negative" in classification:
            return "negative"
        else:
            return "neutral"

    except Exception as e:
        # --- Fallback 2: If the Gemini API call fails ---
        print(f"⚠️ Gemini sentiment API error: {e}. Falling back to VADER.")
        return get_vader_classification(reviews_text)
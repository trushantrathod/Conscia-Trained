import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Download VADER Lexicon (if needed on a new machine) ---
# This checks if the necessary NLTK data is downloaded and gets it if not.
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER lexicon for sentiment analysis (one-time setup)...")
    nltk.download('vader_lexicon')

def get_sentiment_score(reviews_text):
    """
    Calculates a public sentiment score for a given text of reviews.
    The score is normalized to a 0-100 scale for easy display.
    """
    if not reviews_text or pd.isna(reviews_text):
        return 50.0 # Return a neutral score if there are no reviews

    sid = SentimentIntensityAnalyzer()
    
    # VADER's compound score ranges from -1 (most negative) to +1 (most positive).
    compound_score = sid.polarity_scores(str(reviews_text))['compound']
    
    # Normalize the score to a 0-100 scale.
    # (-1 -> 0), (0 -> 50), (+1 -> 100)
    normalized_score = (compound_score + 1) / 2 * 100
    
    return round(normalized_score, 2)


import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

def generate_sentiment_scores():
    """
    Reads the combined product data, calculates a public sentiment score for the reviews
    of each product, and saves the enriched data to a new file.
    """
    # --- Setup Paths ---
    # Get the absolute path of the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, 'data')
    input_filepath = os.path.join(data_folder, 'all_products.csv')
    output_filepath = os.path.join(data_folder, 'products_with_scores.csv')

    # --- Download VADER Lexicon (if needed) ---
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("Downloading VADER lexicon for sentiment analysis (one-time setup)...")
        nltk.download('vader_lexicon')

    # --- Load Data ---
    try:
        df = pd.read_csv(input_filepath)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_filepath}'.")
        print("Please make sure you have run the 'prepare_data.py' script first.")
        return

    # --- Initialize Sentiment Analyzer ---
    sia = SentimentIntensityAnalyzer()

    print("\nGenerating public sentiment scores...")

    # --- Scoring Logic ---
    # This function calculates the sentiment score for a single review text
    def calculate_sentiment(review_text):
        if pd.isna(review_text):
            return 50 # Return a neutral score for missing reviews
        
        # VADER's compound score ranges from -1 (most negative) to +1 (most positive).
        compound_score = sia.polarity_scores(str(review_text))['compound']
        
        # Normalize the score to a 0-100 scale for easier display in the frontend.
        # (-1 -> 0), (0 -> 50), (+1 -> 100)
        normalized_score = (compound_score + 1) / 2 * 100
        return round(normalized_score, 2)

    # Apply the function to the 'reviews' column to create the new score column
    df['public_sentiment_score'] = df['reviews'].apply(calculate_sentiment)
    
    # --- Save Results ---
    df.to_csv(output_filepath, index=False)

    print("\nScoring process complete! ✨")
    print(f"Saved data with sentiment scores to '{output_filepath}'.")
    print(f"Total products scored: {len(df)}")
    print("\nColumns in the final file:")
    print(df.columns.tolist())


if __name__ == '__main__':
    generate_sentiment_scores()
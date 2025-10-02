import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import numpy as np
import nltk

def setup_nltk():
    """Downloads necessary NLTK data files."""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("Downloading VADER lexicon for sentiment analysis (one-time setup)...")
        nltk.download('vader_lexicon')

# A basic list of stop words for cleaning text
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
    'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
}

def clean_text_for_matching(text):
    """
    A simple text cleaner that lowercases, removes punctuation, and splits
    text into word tokens for keyword matching.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text) # Remove punctuation but keep hyphens
    tokens = [word for word in text.split() if word not in STOP_WORDS]
    return tokens

def train_and_score_model_enhanced():
    """
    Analyzes product reviews using a lightweight cleaning function and flexible
    keyword matching to generate more accurate ethical scores.
    """
    all_products_filepath = 'data/all_products.csv'
    output_filepath = 'data/products_with_scores_enhanced.csv'
    
    try:
        # Create the 'all_products.csv' by merging the source files.
        print("Loading and merging source CSV files...")
        electronics_df = pd.read_csv('data/electronics_merged.csv')
        beauty_df = pd.read_csv('data/beauty_merged.csv')
        fashion_df = pd.read_csv('data/fashion_merged.csv')
        groceries_df = pd.read_csv('data/groceries_merged.csv')

        beauty_df['category'] = 'Beauty'
        electronics_df['category'] = 'Electronics'
        fashion_df['category'] = 'Fashion'
        groceries_df['category'] = 'Groceries'

        df = pd.concat([electronics_df, beauty_df, fashion_df, groceries_df], ignore_index=True)
        df.to_csv(all_products_filepath, index=False)
        print("'all_products.csv' created successfully.")
    except FileNotFoundError:
        print("Error: One of the source CSV files was not found. Please ensure they are in the same directory.")
        return

    setup_nltk() # Ensure VADER is downloaded
    sia = SentimentIntensityAnalyzer()
    
    # Using root words for more flexible matching
    keywords = {
        'environmental_impact': ['eco', 'sustainab', 'recycle', 'green', 'organic', 'biodegradable', 'packag', 'waste', 'plastic', 'earth', 'planet', 'carbon'],
        'labor_rights': ['fair-trade', 'ethic', 'handmade', 'local', 'artisan', 'union', 'wage', 'worker', 'sweatshop', 'child-labor', 'supply-chain'],
        'animal_welfare': ['cruelty-free', 'vegan', 'animal-test', 'plant-based', 'humane', 'leaping-bunny', 'peta'],
        'corporate_governance': ['durable', 'quality', 'long-last', 'reliable', 'transparent', 'b-corp', 'scandal', 'recall', 'lawsuit', 'customer-service', 'support', 'honest', 'deceptive', 'bad', 'poor', 'good', 'terrible', 'disappointed', 'excellent', 'fantastic', 'love', 'hate', 'best', 'worst']
    }
    
    results = []
    print("\nStarting enhanced scoring process...")

    for index, row in df.iterrows():
        review_text = str(row['reviews'])
        individual_reviews = re.split(r'\s*\|\s*', review_text)
        
        pillar_sentiments = {pillar: [] for pillar in keywords}
        general_sentiments = []

        for review in individual_reviews:
            if not review.strip():
                continue
            
            sentence_sentiment = sia.polarity_scores(review)['compound']
            
            if -0.05 < sentence_sentiment < 0.05:
                continue

            clean_tokens = clean_text_for_matching(review)
            
            found_keyword_in_review = False
            for token in clean_tokens:
                for pillar, key_list in keywords.items():
                    for key_root in key_list:
                        if token.startswith(key_root):
                            pillar_sentiments[pillar].append(sentence_sentiment)
                            found_keyword_in_review = True
            
            if not found_keyword_in_review:
                general_sentiments.append(sentence_sentiment)
        
        final_scores = {}
        for pillar in keywords:
            sentiments = pillar_sentiments[pillar]
            if pillar == 'corporate_governance':
                sentiments.extend(general_sentiments)

            if sentiments:
                avg_sentiment = np.mean(sentiments)
                score = (avg_sentiment + 1) * 5
            else:
                score = 5.0 # Neutral score if no relevant reviews found

            final_scores[pillar] = max(0.0, min(10.0, round(score, 1)))

        processed_row = {
            'product_id': row['product_id'],
            'product_name': row['product_name'],
            'product_price': row['product_price'],
            'category': row['category'],
            'reviews': row['reviews'],
            'environmental_impact_score': final_scores['environmental_impact'],
            'labor_rights_score': final_scores['labor_rights'],
            'animal_welfare_score': final_scores['animal_welfare'],
            'corporate_governance_score': final_scores['corporate_governance']
        }
        results.append(processed_row)
        
        if (index + 1) % 1000 == 0:
            print(f" - Processed {index + 1}/{len(df)} products...")

    scored_df = pd.DataFrame(results)
    scored_df.to_csv(output_filepath, index=False)
    
    print(f"\nEnhanced scoring process complete! Processed {len(df)} products.")
    print(f"Saved new scores to '{output_filepath}'.")

if __name__ == '__main__':
    train_and_score_model_enhanced()
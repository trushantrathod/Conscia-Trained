from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
import os
from sentiment_analyzer import get_final_sentiment
import google.generativeai as genai
from lime.lime_text import LimeTextExplainer

# --- CONFIG ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBvo6p9LBruiBYDOFygIbDlw0RXe147xB4")

app = Flask(__name__)
CORS(app)

# --- GLOBALS ---
products_df = None
vectorizer  = None
scaler      = None
model       = None
gemini_model = None
explainer   = None

# Column name mapping: CSV space-names → API underscore-names
COL_RENAME = {
    'environmental impact': 'environmental_impact',
    'labor rights':         'labor_rights',
    'animal welfare':       'animal_welfare',
    'corporate governance': 'corporate_governance',
}

ETHICAL_LABELS = ['environmental_impact', 'labor_rights', 'animal_welfare', 'corporate_governance']


# ==============================
# LOAD RESOURCES
# ==============================
def load_resources():
    global products_df, vectorizer, scaler, model, gemini_model, explainer

    base = os.path.dirname(os.path.abspath(__file__))

    try:
        df = pd.read_csv(os.path.join(base, "data", "products_with_scores.csv"))
        df.rename(columns=COL_RENAME, inplace=True)
        
        # Explicitly cast to 'float64' to prevent pandas.errors.LossySetitemError
        for col in ETHICAL_LABELS:
            if col not in df.columns:
                df[col] = 50.0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(50.0).astype('float64')
            
        if 'public_sentiment_score' not in df.columns:
            df['public_sentiment_score'] = 50.0
        df['public_sentiment_score'] = pd.to_numeric(
            df['public_sentiment_score'], errors='coerce'
        ).fillna(50.0).astype('float64')
        
        products_df = df
        print(f"✅ Products loaded: {len(products_df)} rows")
    except Exception as e:
        print("❌ Error loading dataset:", e)
        return False

    try:
        with open(os.path.join(base, "tfidf_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(base, "score_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        model = tf.keras.models.load_model(os.path.join(base, "ethical_model.keras"))
        print("✅ Models loaded successfully")
    except Exception as e:
        print("⚠ ML Model not loaded (will use default/CSV scores):", e)

    # Gemini Setup
    try:
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel("gemini-2.5-flash")
            print("✅ Gemini ready")
    except Exception as e:
        print("⚠ Gemini initialization failed:", e)
        gemini_model = None

    explainer = LimeTextExplainer(class_names=ETHICAL_LABELS)
    return True


# ==============================
# FAST ML PREDICTION
# ==============================
def predict_ethical_scores(text):
    if not all([model, vectorizer, scaler]):
        return {label: 50.0 for label in ETHICAL_LABELS}

    vec  = vectorizer.transform([str(text)]).toarray()
    pred = model.predict(vec, verbose=0)
    scores = scaler.inverse_transform(pred)[0]

    return {
        label: round(float(np.clip(score, 0, 100)), 2)
        for label, score in zip(ETHICAL_LABELS, scores)
    }


# ==============================
# API ROUTES
# ==============================

@app.route("/api/products", methods=["GET"])
def get_products():
    limit  = int(request.args.get("limit",  48))
    offset = int(request.args.get("offset",  0))
    category = request.args.get("category", "All")

    if category and category != "All":
        filtered_df = products_df[products_df['category'].str.lower() == category.lower()]
    else:
        filtered_df = products_df

    subset = filtered_df.iloc[offset: offset + limit]
    records = subset.to_dict(orient="records")

    for record in records:
        for col in ETHICAL_LABELS:
            val = record.get(col)
            record[col] = 50.0 if pd.isna(val) else round(float(val), 2)
            
        sentiment = record.get('public_sentiment_score', 50.0)
        record['public_sentiment_score'] = 50.0 if pd.isna(sentiment) else round(float(sentiment), 2)

    return jsonify(records)


@app.route("/api/products/<product_id>/reviews", methods=["POST"])
def add_review(product_id):
    global products_df

    data = request.get_json()
    new_review = data.get("review")

    if not new_review:
        return jsonify({"error": "No review provided"}), 400

    product_index = products_df[products_df["product_id"] == product_id].index
    if product_index.empty:
        return jsonify({"error": "Product not found"}), 404

    idx = product_index[0]

    # 1. Append review
    current_reviews = products_df.at[idx, "reviews"]
    if pd.notna(current_reviews) and str(current_reviews).strip():
        updated_reviews = f"{current_reviews} | {new_review}"
    else:
        updated_reviews = new_review
    products_df.at[idx, "reviews"] = updated_reviews

    # 2. Update sentiment score
    base_score = float(products_df.at[idx, "public_sentiment_score"])
    new_sentiment_score = get_final_sentiment(base_score, new_review)
    products_df.at[idx, "public_sentiment_score"] = float(new_sentiment_score)

    # 3. PREVENT SCORE RESET: Predict impact of the NEW review only
    new_review_scores = predict_ethical_scores(new_review)
    
    # Apply a Weighted Moving Average (90% Historical Score, 10% New Review Impact)
    for label, predicted_val in new_review_scores.items():
        old_val = float(products_df.at[idx, label])
        # Nudge the score gently based on the new review rather than overwriting it
        blended_score = (old_val * 0.90) + (float(predicted_val) * 0.10)
        products_df.at[idx, label] = round(blended_score, 2)

    # 4. Save to CSV
    data_path = os.path.join(os.path.dirname(__file__), "data", "products_with_scores.csv")
    save_df = products_df.rename(columns={v: k for k, v in COL_RENAME.items()})
    save_df.to_csv(data_path, index=False)

    # 5. Return updated data
    response_data = products_df.iloc[idx].to_dict()
    # Ensure the returned dict uses the freshly blended scores
    for label in ETHICAL_LABELS:
        response_data[label] = products_df.at[idx, label]

    return jsonify(response_data)


@app.route("/api/explain", methods=["POST"])
def explain():
    data = request.get_json()
    text = data.get("reviews", "")

    if not text:
        return jsonify([])

    try:
        if gemini_model:
            prompt = f"""
            Analyze the following product reviews and identify the top 5 key phrases (1-3 words max) that drive the consumer sentiment. 
            For each phrase, assign a sentiment weight:
            - Negative traits (red flags like 'oily', 'sticky', 'broken', 'cheap material') MUST have a negative float between -0.1 and -0.9.
            - Positive traits (like 'smooth', 'durable', 'fresh', 'good quality') MUST have a positive float between 0.1 and 0.9.
            
            Reviews:
            {text}

            You MUST respond with ONLY a valid JSON array of arrays. Do not include any markdown, backticks, or other text.
            Example format:
            [["smooth texture", 0.45], ["sticky", -0.50], ["pleasant fragrance", 0.25], ["oily", -0.40]]
            """
            response = gemini_model.generate_content(prompt)
            
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            import json
            explanation = json.loads(response_text.strip())
            return jsonify(explanation)
        else:
            return jsonify([["AI Offline", 0.0]])
            
    except Exception as e:
        print("XAI Gemini error:", e)
        return jsonify([["Context Analysis Error", 0.0]])


@app.route('/api/snapshot', methods=['POST', 'OPTIONS'])
def snapshot():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.get_json()
    product_name = data.get("product", "").strip().lower()

    if not product_name:
        return jsonify({"error": "No product provided"}), 400

    product_df = products_df[
        products_df['product_name'].str.lower().str.contains(product_name, na=False)
    ]

    if product_df.empty:
        return jsonify({"error": "Product not found"}), 404

    product = product_df.iloc[0]
    ethical = {label: float(product.get(label, 50.0)) for label in ETHICAL_LABELS}

    try:
        if gemini_model:
            ethical_str = "\n".join([f"- {k.replace('_', ' ').title()}: {v}/100" for k, v in ethical.items()])
            prompt = f"""
            Give a concise ethical snapshot of this product for a conscious consumer.
            Product: {product['product_name']}
            Sentiment Score: {product['public_sentiment_score']}/100
            Ethical Scores:
            {ethical_str}
            
            Keep it strictly to 2 short sentences. Highlight the best ethical dimension and one area of concern.
            """
            response = gemini_model.generate_content(prompt)
            summary  = response.text.strip()
        else:
            summary = "AI model not available."
    except Exception as e:
        print("Snapshot error:", e)
        summary = "Could not generate summary at this time."

    return jsonify({
        "product":        product['product_name'],
        "snapshot":       summary,
        "ethical_scores": ethical
    })


@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return '', 200

    data    = request.get_json()
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"reply": "Hey! I'm Conscia 🤖 — ask me about products, recommendations, or ethical scores."})

    try:
        if gemini_model and products_df is not None:
            msg_lower = message.lower()
            filtered_df = products_df.copy()

            cat_map = {
                'beauty': 'Beauty', 
                'electronic': 'Electronics', 'electronics': 'Electronics',
                'fashion': 'Fashion', 
                'grocery': 'Groceries', 'groceries': 'Groceries'
            }
            
            matched_cat = None
            for keyword, actual_cat in cat_map.items():
                if keyword in msg_lower:
                    matched_cat = actual_cat
                    break

            matched_product_names = []
            for p_name in products_df['product_name'].unique():
                if str(p_name).lower() in msg_lower:
                    matched_product_names.append(p_name)

            if matched_product_names:
                filtered_df = filtered_df[filtered_df['product_name'].isin(matched_product_names)]
            elif matched_cat:
                filtered_df = filtered_df[filtered_df['category'] == matched_cat]

            if any(word in msg_lower for word in ["best", "top", "recommend", "good", "highest"]):
                filtered_df = filtered_df.sort_values(by='public_sentiment_score', ascending=False)
            elif any(word in msg_lower for word in ["worst", "bad", "low", "lowest"]):
                filtered_df = filtered_df.sort_values(by='public_sentiment_score', ascending=True)

            top_results = filtered_df.head(10)
            context_str = "DATABASE RESULTS:\n"

            if not top_results.empty:
                for _, row in top_results.iterrows():
                    avg_eth = (row['environmental_impact'] + row['labor_rights'] + row['animal_welfare'] + row['corporate_governance']) / 4
                    reviews_str = str(row['reviews'])
                    reviews_preview = reviews_str[:400] + "..." if len(reviews_str) > 400 else reviews_str
                    
                    context_str += (
                        f"Product: {row['product_name']} | Category: {row['category']} | Price: ₹{row['product_price']}\n"
                        f"Sentiment: {row['public_sentiment_score']}/100 | Avg Ethical: {avg_eth:.1f}/100\n"
                        f"Reviews: {reviews_preview}\n\n"
                    )
            else:
                context_str = "No specific products found matching the query."

            prompt = f"""
            You are Conscia AI, an expert ethical shopping assistant.
            You have been provided with internal database records below.
            
            {context_str}

            User Query: "{message}"

            STRICT INSTRUCTIONS:
            1. Answer the user's query DIRECTLY using ONLY the database results provided above.
            2. NEVER tell the user to "search", "click", or "browse". YOU must provide the answer directly in the chat.
            3. If they ask for recommendations (e.g., "top 10 beauty products"), list the products from the context with their sentiment scores and prices.
            4. If they ask WHY a product has a low/high score, read the "Reviews" text in the context and summarize the specific complaints or praises mentioned by real users. Be specific.
            5. Do NOT invent or hallucinate products.
            6. Do NOT use Markdown formatting like asterisks (** or *) for bolding or bullet points. Output plain text only.
            """

            response = gemini_model.generate_content(prompt)
            reply = response.text.strip().replace('*', '')
        else:
            reply = "AI chat is offline or database is not loaded. Please check your API configuration."

    except Exception as e:
        print("Chat error:", e)
        reply = "Error generating response from AI. Please try again."

    return jsonify({"reply": reply})


if __name__ == "__main__":
    print("🔥 Starting Conscia Server...")
    if load_resources():
        print("✅ Server Ready. Running on port 5000.")
        app.run(debug=True, port=5000)
    else:
        print("❌ Failed to start: Resources missing.")
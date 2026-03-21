from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
import os
import json
from sentiment_analyzer import get_final_sentiment

# --- Gemini (optional) ---
import google.generativeai as genai
from lime.lime_text import LimeTextExplainer

# --- CONFIG ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDUJMgDO6EAVJ5QC-2BeB6xuM2qHrJgMAU")

app = Flask(__name__)
CORS(app)

# --- GLOBALS ---
products_df = None
vectorizer = None
scaler = None
model = None
gemini_model = None
explainer = None


# ==============================
# LOAD RESOURCES
# ==============================
def load_resources():
    global products_df, vectorizer, scaler, model, gemini_model, explainer

    base = os.path.dirname(os.path.abspath(__file__))

    try:
        products_df = pd.read_csv(os.path.join(base, "data", "products_with_scores.csv"))
        print("✅ Products loaded")
    except Exception as e:
        print("❌ Error loading dataset:", e)
        return False

    try:
        with open(os.path.join(base, "tfidf_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)

        with open(os.path.join(base, "score_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)

        model = tf.keras.models.load_model(os.path.join(base, "ethical_model.keras"))

        print("✅ Model loaded")
    except Exception as e:
        print("⚠ Model not loaded:", e)

    # Gemini
    try:
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel("gemini-flash-latest")
            print("✅ Gemini ready")
    except:
        gemini_model = None

    # LIME
    explainer = LimeTextExplainer(class_names=[
        'environmental impact',
        'labor rights',
        'animal welfare',
        'corporate governance'
    ])

    return True


# ==============================
# MODEL PREDICTION
# ==============================
def predict_ethical_scores(text):
    if not all([model, vectorizer, scaler]):
        return [50, 50, 50, 50]

    vec = vectorizer.transform([str(text)]).toarray()
    pred = model.predict(vec)
    scores = scaler.inverse_transform(pred)

    return [round(np.clip(s, 0, 100), 2) for s in scores[0]]


# ==============================
# API ROUTES
# ==============================

@app.route("/api/products", methods=["GET"])
def get_products():
    return jsonify(products_df.to_dict(orient="records"))


# ==============================
# 🔥 UPDATED REVIEW API
# ==============================
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

    # 1️⃣ Append review
    current_reviews = products_df.at[idx, "reviews"]

    if pd.notna(current_reviews) and str(current_reviews).strip():
        updated_reviews = f"{current_reviews} | {new_review}"
    else:
        updated_reviews = new_review

    products_df.at[idx, "reviews"] = updated_reviews


# 2️⃣ 🔥 FIXED SENTIMENT UPDATE
    base_score = float(products_df.at[idx, "public_sentiment_score"])

    new_score = get_final_sentiment(base_score, new_review)

    products_df.at[idx, "public_sentiment_score"] = new_score

    # 3️⃣ Save back
    data_path = os.path.join(os.path.dirname(__file__), "data", "products_with_scores.csv")
    products_df.to_csv(data_path, index=False)

    print(f"✅ Updated sentiment for {product_id} → {new_score}")

    return jsonify(products_df.iloc[idx].to_dict())


# ==============================
# OPTIONAL: EXPLAIN MODEL
# ==============================
@app.route("/api/explain", methods=["POST"])
def explain():
    data = request.get_json()
    text = data.get("reviews", "")

    def predict(texts):
        vec = vectorizer.transform(texts).toarray()
        return model.predict(vec)

    exp = explainer.explain_instance(text, predict, num_features=5)

    return jsonify(exp.as_list())


@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.get_json()
    message = str(data.get("message", "")).strip().lower()

    if not message:
        return jsonify({
            "reply": "Hey! I'm Conscia 🤖 — ask me about products, recommendations, or reviews."
        })

    # ==============================
    # 🔍 CATEGORY DETECTION (AFTER message)
    # ==============================
    category = None

    if "beauty" in message:
        category = "beauty"
    elif "electronics" in message:
        category = "electronics"
    elif "grocery" in message:
        category = "grocery"
    elif "fashion" in message:
        category = "fashion"

    try:
        # ==============================
        # 🔥 FILTER PRODUCTS BY CATEGORY
        # ==============================
        if category:
            filtered = products_df[
                products_df['category'].str.lower().str.contains(category)
            ]
        else:
            filtered = products_df

        # If empty fallback
        if filtered.empty:
            filtered = products_df

        # ==============================
        # 🔥 GET TOP PRODUCTS
        # ==============================
        top_products = filtered.sort_values(
            by='public_sentiment_score',
            ascending=False
        ).head(5)

        product_context = "\n".join([
            f"{row['product_name']} (Score: {round(row['public_sentiment_score'],2)}, Price: {row['product_price']})"
            for _, row in top_products.iterrows()
        ])

        # ==============================
        # 🔍 MATCH SPECIFIC PRODUCT
        # ==============================
        matched_product = None

        for _, row in products_df.iterrows():
            name = str(row['product_name']).lower()
            if name in message:
                matched_product = row
                break

        product_info = ""

        if matched_product is not None:
            product_info = f"""
            Product: {matched_product['product_name']}
            Score: {round(matched_product['public_sentiment_score'],2)}
            Price: {matched_product['product_price']}
            """

        # ==============================
        # 🤖 GEMINI PROMPT
        # ==============================
        prompt = f"""
        You are Conscia AI, a product recommendation assistant.

        User question:
        {message}

        Relevant product:
        {product_info}

        Available products:
        {product_context}

        Instructions:
        - If a product is mentioned, ALWAYS answer about it
        - Tell if it is good or bad based on score
        - Score > 70 → good
        - Score 40–70 → average
        - Score < 40 → not recommended
        - Be clear and direct
        """

        if gemini_model:
            response = gemini_model.generate_content(prompt)
            reply = response.text.strip()
        else:
            reply = "AI not available"

    except Exception as e:
        print("Chat error:", e)
        reply = "Error generating response"

    return jsonify({"reply": reply})

summary_cache = {}  # 🔥 cache for fast response

@app.route('/api/product-analysis', methods=['POST', 'OPTIONS'])
def analyze_product():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.get_json()
    product_name = data.get("product", "").strip().lower()

    if not product_name:
        return jsonify({"error": "No product provided"}), 400

    # 🔍 SMART SEARCH
    product_df = products_df[
        products_df['product_name'].str.lower().str.contains(product_name)
    ]

    if product_df.empty:
        return jsonify({"error": "Product not found"}), 404

    product = product_df.iloc[0]

    # ==============================
    # 🔥 REAL-TIME SENTIMENT
    # ==============================
    base_score = float(product.get("public_sentiment_score", 50))
    user_review = data.get("review", None)

    sentiment_score = get_final_sentiment(base_score, user_review)

    # ==============================
    # 🧠 RECOMMENDATION
    # ==============================
    if sentiment_score >= 75:
        recommendation = "✅ Recommended"
    elif sentiment_score >= 55:
        recommendation = "👍 Good choice"
    elif sentiment_score >= 40:
        recommendation = "⚠️ Consider"
    else:
        recommendation = "❌ Not recommended"

    # ==============================
    # 🤖 GEMINI SUMMARY (CACHED)
    # ==============================
    try:
        product_key = product['product_name']

        if product_key in summary_cache:
            summary = summary_cache[product_key]
        else:
            if gemini_model:
                prompt = f"""
                Give a short 2-3 line product review summary.

                Product: {product['product_name']}
                Sentiment Score: {sentiment_score}/100

                Keep it simple and natural.
                """

                response = gemini_model.generate_content(prompt)
                summary = response.text.strip()
            else:
                summary = "Summary not available."

            summary_cache[product_key] = summary

    except Exception as e:
        print("Gemini error:", e)
        summary = "Could not generate summary."

    # ==============================
    # 💸 BETTER PRODUCTS
    # ==============================
    same_category = products_df[
        products_df['category'] == product['category']
    ]

    better_products = same_category[
        (same_category['public_sentiment_score'] > sentiment_score) &
        (same_category['product_price'] <= product['product_price'])
    ]

    better_products = better_products.sort_values(
        by=['public_sentiment_score', 'product_price'],
        ascending=[False, True]
    ).head(3)

    better_list = better_products[
        ['product_name', 'product_price', 'public_sentiment_score']
    ].to_dict(orient='records')

    # ==============================
    # ✅ FINAL RESPONSE
    # ==============================
    return jsonify({
        "product_name": product['product_name'],
        "sentiment_score": round(sentiment_score, 2),
        "summary": summary,
        "recommendation": recommendation,
        "better_options": better_list
    })

if __name__ == "__main__":
    print("🔥 Starting server...")

    if load_resources():
        print("✅ Resources loaded")
        app.run(debug=True)
    else:
        print("❌ Failed to load resources")
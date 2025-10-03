import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import numpy as np
import tensorflow as tf
import os
import json
import time

# --- Local Imports ---
# This line requires the sentiment_analyzer.py file to be in the same directory.
from sentiment_analyzer import get_sentiment_score

# --- Library Imports for Advanced Features ---
import google.generativeai as genai
from lime.lime_text import LimeTextExplainer

# --- Configuration ---
# For security, it's best practice to load API keys from environment variables.
# You can set this variable in your terminal before running the server.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAIcd1VE4y-MVLPCTQyMz02Mgpty4ukwBo")

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Global Variables for Models & Data ---
products_df = None
vectorizer = None
scaler = None
model = None
gemini_model = None
explainer = None

# --- Core Loading and Prediction Functions ---

def load_resources():
    """Loads all data, models, and necessary assets into memory on startup."""
    global products_df, vectorizer, scaler, model, gemini_model, explainer
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Dataset
    try:
        data_path = os.path.join(script_dir, 'data', 'products_with_scores.csv')
        products_df = pd.read_csv(data_path)
        products_df['product_price'] = pd.to_numeric(products_df['product_price'], errors='coerce')
        print("✅ Successfully loaded product data.")
    except Exception as e:
        print(f"❌ FATAL: Could not load product CSV. {e}")
        return False

    # 2. Load ML Assets
    try:
        vectorizer_path = os.path.join(script_dir, 'tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f: vectorizer = pickle.load(f)
        
        scaler_path = os.path.join(script_dir, 'score_scaler.pkl')
        with open(scaler_path, 'rb') as f: scaler = pickle.load(f)

        model_path = os.path.join(script_dir, 'ethical_model.keras')
        model = tf.keras.models.load_model(model_path)
        print("✅ Successfully loaded custom AI model and assets.")
    except Exception as e:
        print(f"❌ FATAL: Could not load ML assets. {e}")
        return False
        
    # 3. Configure Gemini API
    try:
        if not GEMINI_API_KEY:
            print("⚠️ WARNING: Gemini API Key not found. Chatbot and Summaries will be disabled.")
            gemini_model = None
        else:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
            print("✅ Successfully configured Gemini API.")
    except Exception as e:
        print(f"⚠️ WARNING: Could not configure Gemini API. {e}")
        gemini_model = None
    
    # 4. Initialize LIME Explainer
    class_names = ['environmental impact', 'labor rights', 'animal welfare', 'corporate governance']
    explainer = LimeTextExplainer(class_names=class_names)
    print("✅ Successfully initialized LIME explainer.")
    
    return True

def predict_ethical_scores(text):
    """Predicts the four ethical scores from text using the trained AI model."""
    if not all([model, vectorizer, scaler]):
        return [50.0] * 4

    vectorized_text = vectorizer.transform([str(text)]).toarray()
    predicted_scaled = model.predict(vectorized_text)
    predicted_scores = scaler.inverse_transform(predicted_scaled)
    
    return [round(np.clip(score, 0, 100), 2) for score in predicted_scores[0]]

# --- Chatbot Toolkit ---

def get_product_details(product_name):
    """Fetches the current details of a single product from the DataFrame."""
    product_series = products_df[products_df['product_name'].str.lower() == product_name.lower()]
    if product_series.empty: 
        return {"error": f"I couldn't find a product named '{product_name}'. Please check the spelling."}
    return product_series.iloc[0].to_dict()

def get_recommendations(category, top_n=5):
    """Gets the top N most ethical products in a category."""
    category_df = products_df[products_df['category'].str.lower() == category.lower()].copy()
    if category_df.empty: 
        return {"error": f"Sorry, I don't have a '{category}' category."}
    
    score_cols = ['environmental impact', 'labor rights', 'animal welfare', 'corporate governance']
    category_df['avg_ethical_score'] = category_df[score_cols].mean(axis=1)
    
    top_products = category_df.nlargest(top_n, 'avg_ethical_score')
    return top_products[['product_name', 'avg_ethical_score']].to_dict(orient='records')

def compare_products(product_name_a, product_name_b):
    """Compares the ethical scores of two products."""
    product_a = get_product_details(product_name_a)
    product_b = get_product_details(product_name_b)
    if "error" in product_a or "error" in product_b:
        return {"error": "One or both products could not be found. Please check the names."}
    
    score_cols = ['environmental impact', 'labor rights', 'animal welfare', 'corporate governance']
    return {
        product_a['product_name']: {k: v for k, v in product_a.items() if k in score_cols},
        product_b['product_name']: {k: v for k, v in product_b.items() if k in score_cols}
    }

# --- API Endpoints ---

@app.route('/api/products', methods=['GET'])
def get_products_endpoint():
    """Returns the full list of products and their current scores."""
    if products_df is not None:
        return jsonify(products_df.to_dict(orient='records'))
    return jsonify({"error": "Products not loaded"}), 500

@app.route('/api/products/<product_id>/reviews', methods=['POST'])
def add_review_endpoint(product_id):
    """Adds a new review and dynamically updates the product's scores using the AI model."""
    global products_df
    data = request.get_json()
    new_review = data.get('review')
    if not new_review:
        return jsonify({'error': 'Review content is missing'}), 400

    product_index = products_df[products_df['product_id'] == product_id].index
    if product_index.empty:
        return jsonify({'error': 'Product not found'}), 404
    idx = product_index[0]

    # --- Update Reviews and All Scores ---
    current_reviews = products_df.at[idx, 'reviews']
    updated_reviews = f"{current_reviews} | {new_review}" if pd.notna(current_reviews) else new_review
    products_df.at[idx, 'reviews'] = updated_reviews

    new_ethical_scores = predict_ethical_scores(updated_reviews)
    products_df.at[idx, 'environmental impact'] = new_ethical_scores[0]
    products_df.at[idx, 'labor rights'] = new_ethical_scores[1]
    products_df.at[idx, 'animal welfare'] = new_ethical_scores[2]
    products_df.at[idx, 'corporate governance'] = new_ethical_scores[3]
    
    products_df.at[idx, 'public_sentiment_score'] = get_sentiment_score(updated_reviews)

    data_path = os.path.join(os.path.dirname(__file__), 'data', 'products_with_scores.csv')
    products_df.to_csv(data_path, index=False)
    
    print(f"✅ Updated scores for {product_id} based on new review.")
    
    updated_product = products_df.iloc[idx].to_dict()
    return jsonify(updated_product)


@app.route('/api/snapshot', methods=['POST'])
def get_snapshot_endpoint():
    """Generates a brief summary of a product's scores using Gemini."""
    if not gemini_model: return jsonify({"summary": "Ethical Snapshot feature is currently disabled."})
    
    data = request.get_json()
    product_name = data.get("product_name")
    scores = data.get("scores")
    
    prompt = f"""You are an ethical shopping assistant. Write a short, 2-3 sentence summary for the product '{product_name}' based on these scores (out of 100):
    - Environmental Impact: {scores.get('environmental impact', 'N/A'):.0f}
    - Labor Rights: {scores.get('labor rights', 'N/A'):.0f}
    - Animal Welfare: {scores.get('animal welfare', 'N/A'):.0f}
    - Corporate Governance: {scores.get('corporate governance', 'N/A'):.0f}
    Praise high scores (80+) and mention low scores (below 50) as potential concerns. Be concise and helpful."""
    
    try:
        response = gemini_model.generate_content(prompt)
        return jsonify({"summary": response.text})
    except Exception as e:
        return jsonify({"summary": f"Could not generate summary: {e}"})

@app.route('/api/explain', methods=['POST'])
def explain_prediction_endpoint():
    """Uses LIME to explain which words influenced the model's scores."""
    data = request.get_json()
    review_text = data.get('reviews', '')
    
    def lime_predict_function(texts):
        vectorized = vectorizer.transform(texts).toarray()
        return model.predict(vectorized)

    explanation = explainer.explain_instance(review_text, lime_predict_function, num_features=5, labels=(0, 1, 2, 3))
    
    explanation_data = {
        'environmental impact': [word for word, weight in explanation.as_list(label=0) if weight > 0],
        'labor rights': [word for word, weight in explanation.as_list(label=1) if weight > 0],
        'animal welfare': [word for word, weight in explanation.as_list(label=2) if weight > 0],
        'corporate governance': [word for word, weight in explanation.as_list(label=3) if weight > 0],
    }
    return jsonify(explanation_data)


@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """The main chatbot endpoint that orchestrates calls to the toolkit."""
    if not gemini_model: return jsonify({"reply": "I'm sorry, the AI assistant is currently offline."})
    
    data = request.get_json()
    if not data or 'history' not in data: return jsonify({"reply": "Invalid request format."})
    
    history = data.get('history')
    user_query = history[-1]['parts'][0]['text'] if history else ""

    system_prompt = """
    You are an AI orchestrator. Your job is to determine which tool to call based on the user's query and the conversation history.
    You must respond ONLY with a JSON object containing a "tool" and "parameters".
    If the user's query does not specify a number for `top_n`, you should default to `top_n=3`.
    If the user is just greeting, making small talk, or the query doesn't fit any other tool, you MUST use the `general_knowledge` tool.

    Your available tools are:
    1. `get_product_details(product_name: str)`
    2. `get_recommendations(category: str, top_n: int = 5)`
    3. `compare_products(product_name_a: str, product_name_b: str)`
    4. `general_knowledge()`
    """

    try:
        chat_session = gemini_model.start_chat(history=history[:-1])
        intent_response = chat_session.send_message(f"User Query: \"{user_query}\"\n\n{system_prompt}")
        
        raw_text = intent_response.text.strip().replace("```json", "").replace("```", "").strip()
        
        try:
            intent_data = json.loads(raw_text)
        except json.JSONDecodeError:
            print(f"⚠️ Gemini did not return valid JSON for intent. Defaulting to general_knowledge. Response: {raw_text}")
            intent_data = {"tool": "general_knowledge", "parameters": {}}

        tool_to_call = intent_data.get("tool")
        parameters = intent_data.get("parameters", {})

        tool_map = {
            "get_product_details": get_product_details,
            "get_recommendations": get_recommendations,
            "compare_products": compare_products,
        }

        if tool_to_call in tool_map:
            tool_result = tool_map[tool_to_call](**parameters)
        else:
            tool_result = {"query": user_query}

        response_prompt = f"""
        You are Conscia, a friendly and helpful AI shopping assistant. Formulate a natural, conversational response based on the tool's result.
        - NEVER just output raw JSON.
        - Format lists of products or comparisons clearly with bullet points or tables.
        - If a tool returns an error, explain it to the user in a friendly way.

        TOOL RESULT: {json.dumps(tool_result, indent=2)}

        Now, write your friendly reply to the user.
        """
        final_response = chat_session.send_message(response_prompt)
        return jsonify({"reply": final_response.text})

    except Exception as e:
        print(f"❌ Chatbot error: {e}")
        return jsonify({"reply": "I'm sorry, I had a little trouble processing that. Could you please rephrase your question?"})

# --- Main Execution ---
if __name__ == '__main__':
    if load_resources():
        print("\n✨ All resources loaded. Starting Conscia Universal Server... ✨")
        app.run(debug=True, port=5000)
    else:
        print("\n❌ Failed to load critical resources. Server will not start.")


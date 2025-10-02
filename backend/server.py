import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import os
import google.generativeai as genai
import time
import lime
from lime.lime_text import LimeTextExplainer
import json
import re

GEMINI_API_KEY = "AIzaSyB4piMziCBtdit9FiIRkFjaoo8UKbu28I8"

app = Flask(__name__)
CORS(app) 

products_df = None
vectorizer = None
model = None
gemini_model = None
explainer = None
factual_df = None

def load_resources():
    global products_df, vectorizer, model, gemini_model, explainer, factual_df
    try:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'products_with_scores_enhanced.csv')
        products_df = pd.read_csv(data_path)
        products_df['product_price'] = pd.to_numeric(products_df['product_price'], errors='coerce')
        products_df['product_id'] = products_df['product_id'].astype(str)
        print("Successfully loaded product sentiment data.")
    except Exception as e:
        print(f"Error loading product CSV: {e}")
        return False

    try:
        beauty_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'beauty_merged.csv'))
        electronics_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'electronics_merged.csv'))
        fashion_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'fashion_merged.csv'))
        groceries_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'groceries_merged.csv'))
        factual_df = pd.concat([beauty_df, electronics_df, fashion_df, groceries_df], ignore_index=True)
        factual_df['product_id'] = factual_df['product_id'].astype(str)
        print("Successfully loaded and merged factual score data.")
    except Exception as e:
        print(f"Error loading factual score CSVs: {e}")
        return False

    try:
        vectorizer_path = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print("Successfully loaded TF-IDF vectorizer.")
    except Exception as e:
        print(f"Error loading vectorizer: {e}")
        return False

    try:
        model_path = os.path.join(os.path.dirname(__file__), 'ethical_model.keras')
        model = load_model(model_path)
        print("Successfully loaded custom-trained neural network.")
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return False

    try:
        if not GEMINI_API_KEY:
            print("WARNING: Gemini API Key not found. Chatbot and Summaries will be disabled.")
            gemini_model = None
        else:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-pro')
            print("Successfully configured Gemini API.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        gemini_model = None
    
    explainer = LimeTextExplainer(class_names=['environmental_impact', 'labor_rights', 'animal_welfare', 'corporate_governance'])
    print("Successfully initialized LIME explainer.")
    return True

def get_product_details(product_name):
    product_series = products_df[products_df['product_name'].str.lower() == product_name.lower()]
    if product_series.empty: return {"error": "Product not found."}
    product_data = product_series.iloc[0].to_dict()
    scores = {
        "Environmental": f"{product_data['environmental_impact_score']:.1f}",
        "Labor Rights": f"{product_data['labor_rights_score']:.1f}",
        "Animal Welfare": f"{product_data['animal_welfare_score']:.1f}",
        "Governance": f"{product_data['corporate_governance_score']:.1f}"
    }
    product_data['scores'] = scores
    return product_data

def get_recommendations(category, top_n=5):
    category_df = products_df[products_df['category'].str.lower() == category.lower()].copy()
    if category_df.empty: return {"error": f"Sorry, I don't have a '{category}' category."}
    
    score_columns = ['environmental_impact_score', 'labor_rights_score', 'animal_welfare_score', 'corporate_governance_score']
    category_df['avg_ethical_score'] = category_df[score_columns].mean(axis=1)
    
    top_products = category_df.nlargest(top_n, 'avg_ethical_score')
    return top_products[['product_name', 'avg_ethical_score']].to_dict(orient='records')

def compare_products(product_name_a, product_name_b):
    product_a = get_product_details(product_name_a)
    product_b = get_product_details(product_name_b)
    if "error" in product_a or "error" in product_b:
        return {"error": "One or both products could not be found. Please check the names."}
    return {
        "product_a": {"name": product_a['product_name'], "scores": product_a['scores']},
        "product_b": {"name": product_b['product_name'], "scores": product_b['scores']}
    }

def get_top_expensive_products(category, top_n=5):
    category_df = products_df[products_df['category'].str.lower() == category.lower()].copy()
    if category_df.empty:
        return {"error": f"No products found for category '{category}'."}
    
    top_products = category_df.nlargest(top_n, 'product_price')
    return top_products[['product_name', 'product_price']].to_dict(orient='records')


def get_top_cheapest_products(category, top_n=5):
    category_df = products_df[products_df['category'].str.lower() == category.lower()].copy()
    if category_df.empty:
        return {"error": f"No products found for category '{category}'."}
    
    cheapest_products = category_df.nsmallest(top_n, 'product_price')
    return cheapest_products[['product_name', 'product_price']].to_dict(orient='records')


@app.route('/api/products', methods=['GET'])
def get_products_endpoint():
    if products_df is not None:
        return jsonify(products_df.to_dict(orient='records'))
    else:
        return jsonify({"error": "Products not loaded"}), 500

@app.route('/api/factual-score/<product_id>', methods=['GET'])
def get_factual_score_endpoint(product_id):
    if factual_df is not None:
        product_scores = factual_df[factual_df['product_id'] == str(product_id)]
        if not product_scores.empty:
            score_columns = ['Environmental Impact', 'Labor Rights', 'Animal Welfare', 'Corporate Governance']
            scores = product_scores[score_columns].to_dict(orient='records')[0]
            return jsonify(scores)
        else:
            default_scores = {'Environmental Impact': 50, 'Labor Rights': 50, 'Animal Welfare': 50, 'Corporate Governance': 50}
            return jsonify(default_scores)
    else:
        return jsonify({"error": "Factual scores not loaded"}), 500

@app.route('/api/submit-review', methods=['POST'])
def submit_review_endpoint():
    global products_df
    data = request.get_json()
    product_id, review_text = data.get('product_id'), data.get('review_text')

    if not product_id or not review_text:
        return jsonify({"error": "Product ID and review text are required"}), 400

    product_index_list = products_df.index[products_df['product_id'] == str(product_id)].tolist()
    if not product_index_list:
        return jsonify({"error": "Product not found"}), 404
    
    index = product_index_list[0]

    # Append the new review text to the existing reviews for the session
    # This allows the XAI feature to analyze it immediately on the frontend
    if pd.isna(products_df.loc[index, 'reviews']):
        products_df.loc[index, 'reviews'] = review_text
    else:
        products_df.loc[index, 'reviews'] += f"|||{review_text}"

    # --- Real-time score update logic has been removed ---
    # The new review is accepted, but the scores are not recalculated.
    # This would typically be done in a separate, scheduled batch process.

    return jsonify({
        "message": "Review submitted successfully. Scores are updated periodically."
    })


@app.route('/api/explain', methods=['POST'])
def explain_prediction_endpoint():
    data = request.get_json()
    review_text = data.get('reviews', '')
    
    def lime_predict_function(texts):
        vectorized_texts = vectorizer.transform(texts).toarray()
        return model.predict(vectorized_texts)

    individual_reviews = re.split(r'\|\|\||\. |\? |! ', review_text)
    
    explanations = {}
    for i, single_review in enumerate(filter(None, individual_reviews)):
        explanation = explainer.explain_instance(single_review, lime_predict_function, num_features=5, labels=(0, 1, 2, 3))
        
        explanation_data = {
            'positive': {
                'environmental': [word for word, weight in explanation.as_list(label=0) if weight > 0],
                'labor': [word for word, weight in explanation.as_list(label=1) if weight > 0],
                'animal_welfare': [word for word, weight in explanation.as_list(label=2) if weight > 0],
                'governance': [word for word, weight in explanation.as_list(label=3) if weight > 0],
            },
            'negative': {
                'environmental': [word for word, weight in explanation.as_list(label=0) if weight < 0],
                'labor': [word for word, weight in explanation.as_list(label=1) if weight < 0],
                'animal_welfare': [word for word, weight in explanation.as_list(label=2) if weight < 0],
                'governance': [word for word, weight in explanation.as_list(label=3) if weight < 0],
            }
        }
        explanations[i] = {'review_text': single_review, 'explanation': explanation_data}
        
    return jsonify(explanations)

@app.route('/api/snapshot', methods=['POST'])
def get_snapshot_endpoint():
    if not gemini_model: return jsonify({"summary": "Ethical Snapshot feature is currently disabled."})
    data = request.get_json()
    product_name = data.get("product_name")
    scores = data.get("scores")
    
    scores_100 = {k: v * 10 for k, v in scores.items()}
    
    prompt = f"""You are an ethical shopping assistant. Write a short, 2-3 sentence summary for the product '{product_name}' based on the public sentiment scores (out of 100):
    - Environmental: {scores_100['environmental_impact_score']:.0f}
    - Labor Rights: {scores_100['labor_rights_score']:.0f}
    - Animal Welfare: {scores_100['animal_welfare_score']:.0f}
    - Governance: {scores_100['corporate_governance_score']:.0f}
    Comment on what the public perception seems to be based on these scores. Praise very high scores (80+) and mention low scores (below 50) as potential public concerns. Be concise and helpful."""
    
    response = gemini_model.generate_content(prompt)
    return jsonify({"summary": response.text})

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    if not gemini_model: 
        return jsonify({"reply": "I'm sorry, my connection to the AI assistant is currently offline."})
    
    data = request.get_json()
    if not data or 'history' not in data: 
        return jsonify({"reply": "Invalid request format."})
    
    history = data.get('history')
    user_query = history[-1]['parts'][0]['text'] if history else ""

    system_prompt = """
    You are an AI orchestrator. Your job is to determine which tool to call based on the user's query and the conversation history.
    You must respond ONLY with a JSON object containing a "tool" and "parameters".
    If the user's query does not specify a number for `top_n`, you should default to `top_n=3`.
    If the user is just greeting, making small talk, or the query doesn't fit any other tool, you MUST use the `general_knowledge` tool.

    Your available tools are:
    1. `get_product_details(product_name)`
    2. `get_recommendations(category, top_n=5)`
    3. `compare_products(product_name_a, product_name_b)`
    4. `get_top_expensive_products(category, top_n=5)`
    5. `get_top_cheapest_products(category, top_n=5)`
    6. `general_knowledge()`
    """

    try:
        chat_session = gemini_model.start_chat(history=history[:-1])
        
        intent_response = chat_session.send_message(f"User Query: \"{user_query}\"\n\n{system_prompt}")
        
        raw_text = intent_response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3].strip()
        
        try:
            intent_data = json.loads(raw_text)
        except json.JSONDecodeError:
            print(f"Warning: Gemini did not return valid JSON for intent. Defaulting to general_knowledge. Response was: {raw_text}")
            intent_data = {"tool": "general_knowledge", "parameters": {}}

        tool_to_call = intent_data.get("tool")
        parameters = intent_data.get("parameters", {})

        tool_result = None
        if tool_to_call == "get_product_details":
            tool_result = get_product_details(**parameters)
        elif tool_to_call == "get_recommendations":
            tool_result = get_recommendations(**parameters)
        elif tool_to_call == "compare_products":
            tool_result = compare_products(**parameters)
        elif tool_to_call == "get_top_expensive_products":
            tool_result = get_top_expensive_products(**parameters)
        elif tool_to_call == "get_top_cheapest_products":
            tool_result = get_top_cheapest_products(**parameters)
        else:
            tool_result = {"query": user_query}

        response_prompt = f"""
        You are Conscia, a friendly AI assistant. Formulate a natural, conversational response based on the tool's result, keeping the entire chat history in mind.
        - NEVER just output raw JSON.
        - For lists of products, format them clearly with bullet points using markdown's "-".
        - For comparisons, clearly state which product is better in which category.
        - Be helpful and clear.

        FULL CONVERSATION HISTORY: {json.dumps(history, indent=2)}
        TOOL RESULT FOR LATEST QUERY: {json.dumps(tool_result, indent=2)}

        Now, write your friendly reply to the user.
        """
        final_response = chat_session.send_message(response_prompt)
        return jsonify({"reply": final_response.text})

    except Exception as e:
        if "429" in str(e) and "quota" in str(e).lower():
            print("Chatbot error: Gemini API daily quota exceeded.")
            return jsonify({"reply": "I'm sorry, but I've reached my daily request limit for today. My advanced features will be available again tomorrow. Thank you for your understanding!"})
        else:
            print(f"Chatbot error: {e}")
            return jsonify({"reply": "I'm sorry, I had a little trouble processing that. Could you please rephrase your question?"})


if __name__ == '__main__':
    if load_resources():
        print("All resources loaded. Starting Conscia Universal Server...")
        app.run(debug=True, port=5000)
    else:
        print("Failed to load resources. Server will not start.")
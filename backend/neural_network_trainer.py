import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

print("🚀 Starting Neural Network training process...")

# --- 1. Setup Paths and Load Data ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data', 'products_with_scores.csv')
vectorizer_path = os.path.join(script_dir, 'tfidf_vectorizer.pkl')
scaler_path = os.path.join(script_dir, 'score_scaler.pkl')
model_path = os.path.join(script_dir, 'ethical_model.keras')

try:
    df = pd.read_csv(data_path)
    df['reviews'] = df['reviews'].fillna('') # Handle potential missing reviews
    print(f"✅ Successfully loaded dataset with {len(df)} products.")
except FileNotFoundError:
    print(f"❌ Error: '{data_path}' not found.")
    print("Please run 'prepare_data.py' and 'model_trainer.py' first.")
    exit()

# --- 2. Prepare Data (Features and Labels) ---

# The 'X' is our input feature: the review text.
# The 'y' is our output labels: the four ethical scores we want to predict.
X = df['reviews']
# **Key Change**: Using the correct column names from our prepared data.
y_raw = df[['environmental impact', 'labor rights', 'animal welfare', 'corporate governance']].values

# --- 3. Feature Engineering and Scaling ---

# Vectorize the text data using TF-IDF
print("🔄 Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_vectorized = vectorizer.fit_transform(X).toarray() # Use .toarray() for dense format

# Save the vectorizer to use for future predictions
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"✅ Saved TF-IDF vectorizer to '{vectorizer_path}'.")

# **Best Practice**: Scale the target scores to be between 0 and 1.
# This helps the neural network learn much more effectively.
print("🔄 Scaling target scores...")
scaler = MinMaxScaler()
y = scaler.fit_transform(y_raw)

# Save the scaler to reverse the transformation on future predictions
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✅ Saved score scaler to '{scaler_path}'.")

# --- 4. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
print(f"📊 Data split into {X_train.shape[0]} training and {X_test.shape[0]} testing samples.")

# --- 5. Build and Compile the Neural Network ---
print("🧠 Building the neural network model...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3), # Slightly increased dropout for regularization
    tf.keras.layers.Dense(64, activation='relu'),
    # **Key Change**: The output layer now has a 'sigmoid' activation because we scaled
    # our target scores to be between 0 and 1. Sigmoid is perfect for this range.
    tf.keras.layers.Dense(4, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

model.summary()

# --- 6. Train the Model ---
print("\n🔥 Starting model training...")
history = model.fit(X_train, y_train,
                    epochs=25, # Slightly more epochs for better convergence
                    batch_size=32,
                    validation_split=0.1, # Use a portion of training data for validation
                    verbose=1)

print("\n✅ Training complete!")

# --- 7. Evaluate and Save the Model ---
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print("\n📈 Model performance on test data:")
print(f"   - Loss (Mean Squared Error): {loss:.4f}")
print(f"   - Mean Absolute Error: {mae:.4f}")

model.save(model_path)
print(f"\n✨ Successfully saved the trained model to '{model_path}'.")
print("Ready to be used by the backend server!")
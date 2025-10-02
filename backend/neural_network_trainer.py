import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import matplotlib.pyplot as plt

print("Starting Neural Network training process...")

# --- 1. Load the "Answer Key" Dataset ---
# We use the enhanced dataset we created previously.
try:
    # UPDATE: Changed filename to match the output of the previous script.
    df = pd.read_csv('data/products_with_scores_enhanced.csv')
    df['reviews'] = df['reviews'].fillna('')
    print(f"Successfully loaded dataset with {len(df)} products.")
except FileNotFoundError:
    print("Error: 'products_with_scores_enhanced.csv' not found.")
    print("Please run the previous scoring script first to generate this file.")
    exit()

# --- 2. Prepare the Data for the Neural Network ---
X = df['reviews']
y = df[['environmental_impact_score', 'labor_rights_score', 'animal_welfare_score', 'corporate_governance_score']].values

print("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
# Note: TfidfVectorizer returns a sparse matrix, which is memory-efficient.
X_vectorized = vectorizer.fit_transform(X)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("Saved TF-IDF vectorizer to 'tfidf_vectorizer.pkl'.")

# --- 3. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
# We need to convert the sparse matrices to dense arrays for TensorFlow/Keras
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()
print(f"Data split into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples.")

# --- 4. Build the Neural Network Architecture ---
print("Building the neural network model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3), # Slightly increased dropout for better regularization
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='linear')
])

# --- 5. Compile the Model ---
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])
model.summary()

# --- 6. Train the Model ---
print("\nStarting model training...")
history = model.fit(X_train_dense, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_data=(X_test_dense, y_test),
                    verbose=1)
print("\nTraining complete!")

# --- 7. Evaluate and Save the Model ---
loss, mae = model.evaluate(X_test_dense, y_test, verbose=0)
print(f"\nModel performance on test data:")
print(f"  - Mean Squared Error: {loss:.4f}")
print(f"  - Mean Absolute Error: {mae:.4f}")

model.save('ethical_model.keras')
print("\nSuccessfully saved the trained neural network to 'ethical_model.keras'.")

# --- 8. Visualize Training History (NEW) ---
print("Generating training history plot...")
pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.grid(True)
plt.title('Model Training History')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.gca().set_ylim(0, max(history.history['loss'] + history.history['val_loss'])) # Adjust y-axis for better visibility
plt.savefig('training_history.png')
print("Saved training history plot to 'training_history.png'.")
print("\nProcess complete! You are ready to use this model in your application.")
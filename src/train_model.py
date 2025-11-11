import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_loader import load_data
from preprocess import preprocess_data

# --- CONFIGURATION ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True) # Ensure models folder exists

VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_model.pkl')
# ---------------------

def train():
    print("=== STARTED TRAINING PIPELINE ===")
    
    # 1. Load Data
    df = load_data()
    
    # 2. Preprocess Data (clean the text)
    df = preprocess_data(df)
    
    print("\nSplitting data into training and testing sets...")
    X = df['clean_text'] # The features (text)
    y = df['label']      # The target (0 or 1)
    
    # Split: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Vectorizing text (turning words into numbers)...")
    # TfidfVectorizer converts text to numerical vectors
    # max_features=5000 means we only keep the top 5000 most important words to save memory
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # Learn vocabulary from training data and transform it
    X_train_vec = vectorizer.fit_transform(X_train)
    # Only transform testing data (don't learn from it!)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    print("Training complete!")
    
    # 3. Evaluation
    print("\n=== EVALUATION RESULTS ===")
    y_pred = model.predict(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f} (or {accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 4. Save the Brain (Model + Vectorizer)
    print(f"\nSaving model to {MODEL_DIR}...")
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    print("ALL DONE! Model is ready for use.")

if __name__ == "__main__":
    train()
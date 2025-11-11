import joblib
import os
import sys
from preprocess import clean_text

# Load the saved brain (Vectorizer + Model)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'logistic_model.pkl')

print("Loading model...")
vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

def predict_news(text):
    # 1. Clean the new text just like we did for training
    cleaned_text = clean_text(text)
    
    # 2. Convert it to numbers using the SAME vectorizer
    text_vector = vectorizer.transform([cleaned_text])
    
    # 3. Ask the model for a prediction
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector).max() * 100
    
    label = "REAL NEWS" if prediction == 1 else "FAKE NEWS"
    
    print("\n------------------------------------------------")
    print(f"Result:  [{label}]")
    print(f"Confidence: {probability:.2f}%")
    print("------------------------------------------------\n")

if __name__ == "__main__":
    # If user provides text in command line, use it. Otherwise ask for input.
    if len(sys.argv) > 1:
        user_text = " ".join(sys.argv[1:])
        predict_news(user_text)
    else:
        print("\n--- FAKE NEWS DETECTOR ---")
        user_text = input("Enter a news headline or article snippet:\n> ")
        predict_news(user_text)
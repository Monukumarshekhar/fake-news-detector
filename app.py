import streamlit as st
import joblib
import os
from src.preprocess import clean_text # We re-use our cleaning function

# --- LOAD THE SAVED BRAIN ---
# Get the absolute path to the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define paths to the models
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'logistic_model.pkl')

# Load the vectorizer and model
@st.cache_resource  # This caches the model so it doesn't reload every time
def load_models():
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model

vectorizer, model = load_models()

# --- BUILD THE WEB APP INTERFACE ---

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.write("Enter a news headline or article text below to check if it's real or fake.")
st.write("This model was trained on a dataset of 40,000+ articles and has ~99% accuracy.")

# Create a text area for user input
user_text = st.text_area("Enter news text:", "", height=200)

# Create a button to run the prediction
if st.button("Check News"):
    if user_text:
        # 1. Clean the input text
        cleaned_text = clean_text(user_text)
        
        # 2. Vectorize the text
        text_vector = vectorizer.transform([cleaned_text])
        
        # 3. Predict
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector).max() * 100
        
        # 4. Display the result
        if prediction == 1:
            st.success(f"**Result: REAL NEWS** (Confidence: {probability:.2f}%)")
        else:
            st.error(f"**Result: FAKE NEWS** (Confidence: {probability:.2f}%)")
    else:
        st.warning("Please enter some text to check.")
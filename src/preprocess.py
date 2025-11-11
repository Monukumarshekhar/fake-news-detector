import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK data (only needs to happen once)
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans raw text by removing URLs, special characters, and stopwords.
    """
    # 1. Lowercase everything
    text = str(text).lower()
    
    # 2. Remove URLs (http://...)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove HTML tags (<br>, <html>, etc.)
    text = re.sub(r'<.*?>', '', text)
    
    # 4. Remove punctuation and special characters
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', ' ', text) # remove newlines
    text = re.sub(r'\w*\d\w*', '', text) # remove words containing numbers
    
    # 5. Remove stopwords (common words that don't add much meaning)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

def preprocess_data(df):
    print("Starting text preprocessing... (This might take a minute)")
    # Apply the clean_text function to the 'text' column
    # We create a NEW column called 'clean_text' so we don't lose the original
    df['clean_text'] = df['text'].apply(clean_text)
    print("Preprocessing complete!")
    return df

if __name__ == "__main__":
    # Test run to see if it works
    from data_loader import load_data
    
    # Load a small sample just to test quickly
    df = load_data().head(500) 
    df = preprocess_data(df)
    
    print("\nComparison (Original vs Cleaned):")
    print(df[['text', 'clean_text']].head(2))
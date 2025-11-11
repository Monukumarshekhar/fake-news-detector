# 📰 Fake News Detector

A web application built with Python, scikit-learn, and Streamlit that classifies news articles as "Real" or "Fake". This project demonstrates a complete end-to-end NLP pipeline, from data cleaning and model training to deployment as an interactive web app.

## 🚀 Live Demo

*(This is a placeholder for when we deploy to Streamlit Community Cloud. We'll fill this in later.)*

## 🛠️ Tech Stack

* **Python 3.10+**
* **Scikit-learn:** For TF-IDF Vectorization and Logistic Regression.
* **Pandas:** For data loading and manipulation.
* **NLTK:** For text preprocessing (stopwords).
* **Streamlit:** For building the interactive web app.
* **Joblib:** For saving and loading the trained model.

---

## 🔬 Key Finding: A Lesson in Dataset Bias

This project's most valuable outcome was not the final model, but the discovery of significant dataset bias.

After training, the model achieved **99%+ accuracy** on its test set. However, when tested with new, real-world headlines in the Streamlit app, it failed, classifying almost everything as "FAKE."

### The Root Cause

Upon investigation, I discovered the model wasn't learning the *semantics* of fake vs. real news. Instead, it learned a lazy shortcut:

* The **`True.csv`** file consisted entirely of articles from **Reuters**.
* The model learned that if an article contained the keyword **"(Reuters)"**, it was **REAL**.
* If this keyword was missing (as it is in all other news articles), it was classified as **FAKE**.

This project serves as a critical case study on why a high accuracy score can be misleading and why understanding and validating your data is the most important step in machine learning.

---

## 💻 How to Run This Project Locally

### 1. Clone the Repository

```bash
git clone [https://github.com/Monukumarshekhar/fake-news-detector.git](https://github.com/Monukumarshekhar/fake-news-detector.git)
cd fake-news-detector
```

### 2. Create and Activate a Virtual Environment

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

The `src/train_model.py` script requires `scikit-learn`, `pandas`, and `nltk`. The `app.py` requires `streamlit`. You can install them all with:

```bash
pip install scikit-learn pandas nltk streamlit
```

### 4. Run the Streamlit App

Once everything is installed, run the main application:

```bash
streamlit run app.py
```

This will open the web app in your default browser.

---

## 📁 Project Structure

```
fake-news-detector/
├── data/
│   ├── Fake.csv
│   └── True.csv
├── models/
│   ├── logistic_model.pkl
│   └── tfidf_vectorizer.pkl
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── predict.py
├── .gitignore
├── app.py             # The main Streamlit web app
└── README.md          # You are here
```
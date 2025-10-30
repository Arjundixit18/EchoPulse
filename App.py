import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib

# Make sure stopwords are available
nltk.download('stopwords')

# Load the saved model and vectorizer
model = joblib.load(r"D:\PLACEMENTS\Projects\EchoPulse\model.pkl")
vectorizer = joblib.load(r"D:\PLACEMENTS\Projects\EchoPulse\vectorize.pkl")

# Text cleaning function
def clean_text(text):
    ps = PorterStemmer()
    stop_words = stopwords.words('english')
    text = re.sub('[^a-zA-Z]', ' ', text)   # keep only letters
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# Streamlit UI
st.title("📊 Sentiment Analysis on Tweets")

input_text = st.text_area("Enter a tweet")

if st.button("Analyze"):
    if input_text.strip() != "":
        cleaned = clean_text(input_text)
        vectorized = vectorizer.transform([cleaned]).toarray()
        result = model.predict(vectorized)

        # Map prediction to label
        label_map = {1: "Negative", 2: "Neutral", 3: "Positive"}
        label = label_map.get(result[0], "Unknown")

        st.success(f"Predicted Sentiment: **{label}**")
    else:
        st.warning("⚠️ Please enter some text before analyzing.")

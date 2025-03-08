import streamlit as st
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

# Load trained model & vectorizer
clf = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize text processor
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words("english"))

# Function to predict spam
def predict_spam(email_text):
    processed_text = email_text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    processed_text = [stemmer.stem(word) for word in processed_text if word not in stopwords_set]
    processed_text = " ".join(processed_text)

    x_email = vectorizer.transform([processed_text])
    prediction = clf.predict(x_email)[0]
    return " SPAM!" if prediction == 1 else " NOT Spam."

# Streamlit UI
st.title(" Email Spam Detector")
email_text = st.text_area("Enter email content:")

if st.button("Check Spam"):
    result = predict_spam(email_text)
    st.subheader(result)

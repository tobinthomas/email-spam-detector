import string
import tkinter as tk
from tkinter import messagebox

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords if not available
nltk.download('stopwords')

# Load trained model and vectorizer
try:
    clf = joblib.load("spam_classifier.pkl")  # Load trained spam detection model
    vectorizer = joblib.load("vectorizer.pkl")  # Load text vectorizer
except Exception as e:
    messagebox.showerror("Error", "Model or vectorizer file not found. Train and save them first.")
    exit()

# Initialize Porter Stemmer
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words("english"))

# Function to process and predict email
def predict_spam():
    email_text = email_input.get("1.0", tk.END).strip()
    
    if not email_text:
        messagebox.showerror("Error", "Please enter an email to classify.")
        return
    
    # Preprocess input email
    processed_text = email_text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    processed_text = [stemmer.stem(word) for word in processed_text if word not in stopwords_set]
    processed_text = " ".join(processed_text)

    # Vectorize and predict
    x_email = vectorizer.transform([processed_text])
    prediction = clf.predict(x_email)[0]

    # Display result
    if prediction == 1:
        messagebox.showinfo("Result", "🚨 This email is SPAM!")
    else:
        messagebox.showinfo("Result", "✅ This email is NOT Spam.")

# Create GUI
root = tk.Tk()
root.title("Gmail Spam Detector")
root.geometry("500x350")

tk.Label(root, text="Enter Gmail Content:", font=("Arial", 12)).pack(pady=5)

email_input = tk.Text(root, height=7, width=60)
email_input.pack()

predict_button = tk.Button(root, text="Check Spam", font=("Arial", 12), command=predict_spam)
predict_button.pack(pady=10)

root.mainloop()

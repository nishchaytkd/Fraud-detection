import streamlit as st
import pickle
import string
import re

# Load saved model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# UI
st.title("Spam Detection App")

user_input = st.text_area("Enter your message")

if st.button("Check"):
    cleaned = clean_text(user_input)
    vector_input = vectorizer.transform([cleaned])
    prediction = model.predict(vector_input)

    if prediction[0] == 1:
        st.error("🚨 Spam")
    else:
        st.success("✅ Not Spam")

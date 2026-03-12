import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("svm_sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

def predict_sentiment(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    probs = model.predict_proba(text_tfidf)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"

    return sentiment, probs[1], probs[0]

st.write("## Sentiment Analysis App for Movie Reviews")

user_input = st.text_area("Enter movie review:")

if st.button("Predict"):

    sentiment, pos_prob, neg_prob = predict_sentiment(user_input)

    st.write("### Prediction:", sentiment)
    st.write("### Confidence Score:", round(pos_prob, 2) if sentiment == "Positive" else round(neg_prob, 2))

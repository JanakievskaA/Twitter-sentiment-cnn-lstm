import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

model = load_model('sentiment_model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])
    return text

st.title('Twitter Sentiment Analysis')
st.write('Enter a tweet to predict its sentiment')

tweet = st.text_input('Tweet:')

if st.button('Predict'):
    cleaned = clean_text(tweet)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=64)
    pred = model.predict(padded)
    labels = ['Irrelevant', 'Negative', 'Neutral', 'Positive']
    result = labels[np.argmax(pred)]
    st.success(f'Sentiment: {result}')

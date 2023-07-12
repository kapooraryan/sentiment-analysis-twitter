import streamlit as st
import pandas as pd
import nltk
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import string

# Load the saved model
with open("sent_model.pkl", "rb") as f:
    sent_model = pickle.load(f)

# Download NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')


# Set up stopwords and lemmatizer
words_stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if (word not in words_stop) and (word not in string.punctuation)]
    text = ' '.join(text)
    return text

# Load sentiment data
text_sentiment_data = pd.read_csv('Tweets.csv')
text_sentiment_df = text_sentiment_data.drop(text_sentiment_data[text_sentiment_data['airline_sentiment_confidence'] < 0.5].index, axis=0)
X = text_sentiment_df['text']
Y = text_sentiment_df['airline_sentiment']
sentiments = ['negative', 'neutral', 'positive']
Y = Y.apply(lambda x: sentiments.index(x))
cleaned_text_data = [preprocess_text(text) for text in X]

# Fit the CountVectorizer
count_vectorizer = CountVectorizer(max_features=5000, stop_words=['virginamerica', 'united'])
XFit = count_vectorizer.fit_transform(cleaned_text_data).toarray()

# Train the model
sent_model.fit(XFit, Y)

def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    vectorized_text = count_vectorizer.transform([cleaned_text]).toarray()
    prediction = sent_model.predict(vectorized_text)[0]
    sentiment = sentiments[prediction]
    return sentiment

# Streamlit app
def main():
    st.title("Sentiment Analysis App")
    st.write("Enter a text and get the predicted sentiment.")

    text_input = st.text_input("Text")
    if st.button("Predict"):
        if text_input.strip() != "":
            result = predict_sentiment(text_input)
            st.success(f"Predicted Sentiment: {result}")
        else:
            st.warning("Please enter some text.")

if __name__ == '__main__':
    main()

# %%
#Importing required libraries
import pandas as pd
import nltk

# %%
#Reading the dataset
text_sentiment_data = pd.read_csv('tweets.csv')

# %%
text_sentiment_data.head(10)

# %%
text_sentiment_data.shape

# %%
#Filtering out entries with low confidence
text_sentiment_df = text_sentiment_data.drop(text_sentiment_data[text_sentiment_data['airline_sentiment_confidence']<0.5].index, axis= 0)

# %%
text_sentiment_df.shape

# %%
X = text_sentiment_df['text']
Y = text_sentiment_df['airline_sentiment']

# %%
# Cleaning our text data:

from nltk.corpus import stopwords
nltk.download('stopwords')
import string
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# %%
#Initializing stopwords and punctuation so we can remove them from our text data
words_stop = stopwords.words('english')
punctuations = string.punctuation

# %%
#With the help of regular expressions we will consider only the text part and remove all the numbers, special characters from the text data
import re
nltk.download('wordnet')

#First take text data and convert it to lower case and then lemmatize it (reduce it to its lower form, for exapmle: walked -> walk)
cleaned_text_data = []
for i in range(len(X)):
  text = re.sub('[^a-zA-Z]', ' ',X.iloc[i])
  text = text.lower().split()
  text = [lemmatizer.lemmatize(word) for word in text if (word not in words_stop) and (word not in punctuations)]
  text = ' '.join(text)
  cleaned_text_data.append(text)

# %%
cleaned_text_data

# %%
Y

# %%
# Sentiment labels are converted into numerical representations

sentiments = ['negative' , 'neutral', 'positive']
Y = Y.apply(lambda x: sentiments.index(x))

# %%
Y.head()

# %%
#Vectorizing the text data
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(max_features = 5000, stop_words = ['virginamerica','united'])
XFit = count_vectorizer.fit_transform(cleaned_text_data).toarray()

# %%
XFit.shape

# %%
#Creating the model
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
sent_model = MultinomialNB()

# %%
#Splitting of training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(XFit,Y, test_size = 0.3)

# %%
#Fit the model
sent_model.fit(X_train,Y_train)

# %%
#Evaluating the model
y_pred = sent_model.predict(X_test)

# %%
#Plotting the confusion matrix
from sklearn.metrics import classification_report

classification = classification_report(Y_test,y_pred)
print(classification)

# %%
#Saving the model for deployment
import pickle

with open("sent_model.pkl", "wb") as f:
    pickle.dump(sent_model, f)



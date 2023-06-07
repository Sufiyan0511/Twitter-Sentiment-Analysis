#!/usr/bin/env python
# coding: utf-8

# # Importing Important Libaries

# In[1]:


import pandas as pd
import numpy as np
import re
import spacy
import nltk
import multiprocessing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('C:/Users/Dell/Downloads/Sentiment140.csv', encoding='latin-1', header=None)


# In[3]:


df.columns = ['Sentiment', 'ID', 'Date', 'Flag', 'User', 'Tweet']


# In[4]:


df.shape


# # Data Preprocessing

# In[24]:


# Randomly sample 10,000 observations from your dataset
df = df.sample(n=10000, random_state=42)


# In[25]:


df.head()


# In[26]:


df['Sentiment'].value_counts()


# In[27]:


#!python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')


# In[ ]:


# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('omw-1.4')


# In[28]:


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    
    # Remove mentions
    text = re.sub(r"@\w+", "", text)
    
    # Remove special characters and punctuation
    text = re.sub(r"[^\w\s]", "", text)
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into a string
    preprocessed_text = " ".join(tokens)
    
    return preprocessed_text


# In[29]:


df['Cleaned tweet'] = df['Tweet'].apply(preprocess_text)


# In[30]:


df['Cleaned tweet']


# In[31]:


# Replace Sentiment values
df['Sentiment'] = df['Sentiment'].replace({0: "negative", 4: "positive"})

# Verify the updated Sentiments
print(df['Sentiment'].value_counts())


# # Visualization

# In[32]:


plt.figure(figsize =(8,6))
sns.countplot( x = 'Sentiment', data = sampled_data)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title("Sentiment Distribution")

plt.show()


# In[22]:


# Separate positive and negative tweets
positive_tweets = ' '.join(df[df['Sentiment'] == 'positive']['Cleaned tweet'].values)
negative_tweets = ' '.join(df[df['Sentiment'] == 'negative']['Cleaned tweet'].values)


# In[33]:


# Calculate word frequency
word_frequency = sampled_data['Cleaned tweet'].str.split(expand=True).stack().value_counts()

# Plot the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=word_frequency.index[:10], y=word_frequency.values[:10])
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Frequent Words')
plt.xticks(rotation=45)
plt.show()


# # Preparing The NLP Model

# In[34]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


# In[40]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Cleaned tweet'], df['Sentiment'], test_size=0.2, random_state=42)


# In[41]:


# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# # Naive Bayes Classifier

# In[42]:


# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)


# In[43]:


# Evaluate the model
y_pred = classifier.predict(X_test_vectorized)
print(classification_report(y_test, y_pred))


# # SVM

# In[44]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


# In[45]:


# Train the SVM classifier
classifier = SVC()
classifier.fit(X_train_vectorized, y_train)


# In[46]:


# Evaluate the model
y_pred = classifier.predict(X_test_vectorized)
print(classification_report(y_test, y_pred))


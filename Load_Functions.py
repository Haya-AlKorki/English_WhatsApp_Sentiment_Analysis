import re
import string
import pandas as pd
import numpy as np
from collections import Counter
import Data_Preprocessing , Load_Model
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import BatchEncoding
from PIL import Image,ImageDraw,ImageFont
from wordcloud import WordCloud
import tensorflow as tf
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def cleaned_text(text):
    text = re.sub(r'^.*\bomitted\b.*\n?|Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them', '', text)
    text = re.sub(r'^.*(?:\S+ added you|\S+ added \S+).*\n?', '', text)
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text_clean_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    text = re.sub(text_clean_re, ' ', text).strip()
    return text

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    token = [word for word in text.split() if word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = []
    for t in token:
      lemmatize = lemmatizer.lemmatize(t, wordnet.VERB)
      tokens.append(lemmatize)
    return " ".join(tokens)

def final_cleaned_text(text):
  text = cleaned_text(text)
  text = preprocess_text(text)
  return text

def ct(x):
  x['Cleaned_Text'] = final_cleaned_text(x['Text'])
  return x

def sentiment(d):
    if d['value'] == 2:
        return 'positive'
    if d['value'] == 0:
        return 'negative'
    if d['value'] == 1:
        return 'neutral'

def change_value(d):
    if d == 2:
        return 'positive'
    if d == 0:
        return 'negative'
    if d == 1:
        return 'neutral'

def predict_sentiment(user_input, model, vectorizer):
    vectorized_input = vectorizer.transform([user_input])
    prediction = model.predict(vectorized_input)[0]
    return prediction


def predict_sentiment_for_column(data_column, model, vectorizer):
        vectorized_input = vectorizer.transform(data_column)
        pred = model.predict(vectorized_input)
        return pred

def fun_stats(selected_user, df):
    if selected_user != "Overall":
        df = df[df['User'] == selected_user]
    total_messages = df.shape[0]
    total_words = []
    for message in df['Cleaned_Text']:
        total_words.extend(message.split())
    return total_messages, len(total_words)

def most_busy_users(data):
    x = data['User'].value_counts().head()
    data2 = round((data['User'].value_counts() / data.shape[0]) * 100, 2).reset_index().rename(columns={'index': 'name', 'User': 'percetage'})
    return x,data2

def month_activity_map(selected_user,df,k):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]
    df = df[df['value'] == k]
    return df['Month'].value_counts()

def week_activity_map(selected_user,df,k):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]
    df = df[df['value'] == k]
    return df['Day_name'].value_counts()

def activity_heatmap(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]
    df = df[df['value'] == k]
    user_heatmap = df.pivot_table(index='Day_name', columns='period', values='Cleaned_Text', aggfunc='count').fillna(0)
    return user_heatmap

def daily_timeline(selected_user,df,k):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]
    df = df[df['value']==k]
    # count of message on a specific date
    daily_timeline = df.groupby('Only_Date').count()['Cleaned_Text'].reset_index()
    return daily_timeline

def monthly_timeline(selected_user,df,k):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]
    df = df[df['value']==k]
    timeline = df.groupby(['Year', 'Month_num', 'Month']).count()['Cleaned_Text'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['Month'][i] + "-" + str(timeline['Year'][i]))
    timeline['time'] = time
    return timeline

def percentage(df,k):
    df = round((df['User'][df['value']==k].value_counts() / df[df['value']==k].shape[0]) * 100, 2).reset_index().rename(columns={'index': 'name', 'User': 'percent'})
    return df

def most_busy_users(df):
    x = df['User'].value_counts().head()
    df = round((df['User'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'User': 'percent'})
    return x, df

def pie_chart(data):
  red = [(0.8901960784313725, 0.10196078431372549, 0.10980392156862745)]
  orange =[(1.0, 0.4980392156862745, 0.0)]
  green =[(0.2, 0.6274509803921569, 0.17254901960784313)]
  labels = ["positive", "neutral", "negative"]
  plt.pie(data , labels = labels, colors=green+orange+red, autopct='%.0f%%')
  plt. show()

def get_word_cloud(self):
    data = ' '.join(self.Cleaned_Text)
    wordcloud = WordCloud(font_path='arial', max_font_size=80, max_words=50, background_color="white").generate(data)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
def get_word_cloud_negative(self):
    data = ' '.join(self[self['Sentiment']=='negative'].Cleaned_Text)
    wordcloud = WordCloud(font_path='arial', max_font_size=80, max_words=50, background_color="white").generate(data)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
def get_word_cloud_positive(self):
    data = ' '.join(self[self['Sentiment']=='positive'].Cleaned_Text)
    wordcloud = WordCloud(font_path='arial', max_font_size=80, max_words=50, background_color="white").generate(data)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def get_word_cloud_neutral(self):
    data = ' '.join(self[self['Sentiment']=='neutral'].Cleaned_Text)
    wordcloud = WordCloud(font_path='arial', max_font_size=80, max_words=50, background_color="white").generate(data)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


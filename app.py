import streamlit as st
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
import os

import config
from modules import *
# Authenticate
auth = tweepy.OAuthHandler(config.twitterApiKey, config.twitterApiSecret)
auth.set_access_token(config.twitterApiAccessToken, config.twitterApiAccessTokenSecret)
twetterApi = tweepy.API(auth, wait_on_rate_limit = True)



st.title('Sentiment Analysis of a users tweets')
st.markdown('Enter user and number of tweets')
twitterAccount = st.text_input('Twitter User','elonmusk')
n = int(st.text_input('Number of tweets','50'))

tweets = tweepy.Cursor(twetterApi.user_timeline, 
                    screen_name=twitterAccount, 
                    count=None,
                    since_id=None,
                    max_id=None,
                    trim_user=True,
                    exclude_replies=True,
                    contributor_details=False,
                    include_entities=False
                    ).items(n);

df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweet'])
st.dataframe(df)

st.text('Processing Data...')

df['Tweet'] = df['Tweet'].apply(cleanUpTweet)
df['Subjectivity'] = df['Tweet'].apply(getTextSubjectivity)
df['Polarity'] = df['Tweet'].apply(getTextPolarity)
df = df.drop(df[df['Tweet'] == ''].index)
df['Score'] = df['Polarity'].apply(getTextAnalysis)
positive = df[df['Score'] == 'Positive']
labels = df.groupby('Score').count().index.values
values = df.groupby('Score').size().values
plt.bar(labels, values)

for index, row in df.iterrows():
    if row['Score'] == 'Positive':
        plt.scatter(row['Polarity'], row['Subjectivity'], color="green")
    elif row['Score'] == 'Negative':
        plt.scatter(row['Polarity'], row['Subjectivity'], color="red")
    elif row['Score'] == 'Neutral':
        plt.scatter(row['Polarity'], row['Subjectivity'], color="blue")

plt.title('Twitter Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.savefig('fig1.png')
# add legend
st.image('fig1.png')

# Creating a word cloud
words = ' '.join([tweet for tweet in df['Tweet']])
wordCloud = WordCloud(width=600, height=400).generate(words)

plt.imshow(wordCloud)
plt.savefig('fig2.png')
st.image('fig2.png')
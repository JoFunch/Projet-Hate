import pandas as pd
import numpy as np
import nltk
import re
#find danish stopwords

import lemmy
import seaborn as sb

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



stopword = stopwords.words('danish')

lemmatizer = lemmy.load("da")


#sb.countplot('subtask_a', data=hate_df)


def preprocess_text ( text ):

	text = str(text).lower()
	text = re.sub('\[.*?\]', '', text)
	text = re.sub('https?://\S+|www\.\S+', '', text)
	text = re.sub('<.*?>+', '', text)
	#text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
	text = re.sub('\n', '', text)
	text = re.sub('[.,?\/]', '', text)
	text = re.sub('\w*\d\w*', '', text)
	text = [word for word in text.split(' ') if word not in stopword]
	text = " ".join(text)
	text = [lemmatizer.lemmatize('', word)[0] for word in text.split(' ')]
	text = " ".join(text)
	return text



def load_data(tsv):
	"""
	to take a csv-file (here tsv) and return a normalised DF for further processing.
	The DF must incl.
			
		preprocessing of tweets
			removal of symbols, hashtags, and smilies
			removal of websites,
			removal of punctuation
			tokenisation, stemming/normalisation of words
		Numerical/array of tweets
		label-encoded target-values


	"""
	hate_df = pd.read_csv(tsv, delimiter="\t", header=[0])

	hate_df['tweet'] = hate_df['tweet'].apply(preprocess_text)

	return hate_df['tweet'], hate_df['subtask_a']


hate_df = load_data('dkhate/oe20da_data/offenseval-da-training-v1.tsv')


def split(x, y):
	x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, shuffle=True)



	count = CountVectorizer(ngram_range=(1,3))
	tfidf = TfidfTransformer()

	x_train_vectorizer=count.fit_transform(x_train)
	x_test_vectorizer=count.fit_transform(x_test)

	x_train_tfidf = tfidf.fit_transform(x_train_vectorizer)
	x_train_tfidf = x_train_tfidf.toarray()

	x_test_tfidf = tfidf.fit_transform(x_test_vectorizer)
	x_test_tfidf = x_test_tfidf.toarray()

	return x_train_tfidf, x_test_tfidf, y_train, y_test


X_train, X_test, y_train, y_test = split(hate_df[0], hate_df[1])



#make tensors
#DataLoader
#Neural Network
#Train
#Eval






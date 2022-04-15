import pandas as pd
import numpy as np
import nltk
import re
#find danish stopwords

import string
import lemmy
import seaborn as sb

import xlrd


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing

import matplotlib.pyplot as plt

from collections import Counter


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_multilabel_classification


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.autograd import Variable

from torchtext.data.functional import sentencepiece_numericalizer
from afinn import Afinn
afinn = Afinn(emoticons=True)

import random
import torch.nn.functional as F

stopword = stopwords.words('danish')

lemmatizer = lemmy.load("da")

def preprocess_text ( text ):

	text = str(text)
	return text

def preprocess_text1 ( text ):

	text = str(text).lower()
	text = re.sub(r'\[.*?\,\/]', '', text)
	text = re.sub(r'https?://\S+|www\.\S+', '', text) #websites
	text = re.sub(r'<.*?>+', '', text) #links, urls, etc. web/twitter formats
	text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
	text = re.sub(r'\n', '', text)
	text = re.sub(r'\w*\d\w*', '', text)
	text = re.sub(r'(.)\1{2}', '', text) #remove < 2 (meaning 3) occurrences of same characters
	text = re.sub(r'\s\s', ' ', text)
	text = [word for word in text.split(' ')]# if word not in stopword] #stopwords
	text = " ".join(text)
	#text = [lemmatizer.lemmatize('', word)[0] for word in text.split(' ')] #lemmatization
	#text = " ".join(text) 
	return text


def split(x, y):

	x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=16, shuffle=True)

	x_train = np.array(x_train)
	y_train = np.array(y_train)

	x_test = np.array(x_test)
	y_test = np.array(y_test)

	
	x_train_tensors = torch.Tensor(x_train)
	y_train_tensors = torch.Tensor(y_train)

	x_test_tensors = torch.Tensor(x_test)
	y_test_tensors = torch.Tensor(y_test)


	x_train_tensors_final = torch.reshape(x_train_tensors,  (x_train_tensors.shape[0], 1, x_train_tensors.shape[1]))
	x_test_tensors_final = torch.reshape(x_test_tensors,  (x_test_tensors.shape[0], 1, x_test_tensors.shape[1]))

	return x_train_tensors_final, x_test_tensors_final, y_train_tensors, y_test_tensors #, x_train, x_test, y_train, y_test

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
	#Load sentiment
	hate_df = pd.read_csv(tsv, delimiter=';', header=[0])
	hate_df = hate_df.fillna('None')

	for index,data in hate_df.iterrows():
		if len(word_tokenize(data['Text'])) < 4 or len(word_tokenize(data['Text'])) > 70:
			hate_df.drop(index, inplace=True)
	

	#labels, A, B, C

	subtask_a = hate_df['Sub-Task A']
	subtask_b = hate_df['Sub-Task B'].tolist()
	subtask_c = hate_df['Sub-Task C'].tolist()


	suba = preprocessing.LabelEncoder()
	subtask_a_encode = suba.fit(subtask_a)
	subtask_a_encode = suba.transform(subtask_a)
	
	subb = preprocessing.LabelEncoder()
	subtask_b_encode = subb.fit(subtask_b)
	subtask_b_encode = subb.transform(subtask_b)

	subc = preprocessing.LabelEncoder()
	subtask_c_encode = subc.fit(subtask_c)
	subtask_c_encode = subc.transform(subtask_c)

	#Classes and inverse!
	print(subc.classes_)
	
	
	hate_tweets = hate_df['Text']


	
	
	hate_tweets = hate_tweets.apply(preprocess_text1)
	counts = Counter()
	for row in hate_tweets:
		counts.update(word_tokenize(row))

	words = 0
	for word in counts:
		words += 1


	hate_tweets = pd.DataFrame(hate_tweets, columns = ['Text'])


	sentiments = hate_df['Text'].tolist()
	scored_sentences = [afinn.score(sent) for sent in sentiments]
	scored_sentences = pd.DataFrame(scored_sentences)

	#create label-dictionary, unnest lists, padding, scaling, concat
	tokenizer = Tokenizer()
	# MinMax = MinMaxScaler()
	Scaler = StandardScaler()

	tokenized = hate_tweets['Text'].apply(lambda X: tokenizer.fit_on_texts(X))
	
	hate_tweets['Text'] = hate_tweets['Text'].apply(lambda X: tokenizer.texts_to_sequences(X))
	hate_tweets['Text'] = hate_tweets['Text'].apply(lambda X: [float(item) for items in X for item in items]) #change for float?
	hate_tweets['Text'] = hate_tweets['Text'].apply(lambda X: pad_sequences([X], maxlen=69)) #change padlen
	hate_tweets['Text'] = hate_tweets['Text'].apply(lambda X: [float(item) for items in X for item in items])
	#hate_tweets['Text'] = hate_tweets['Text'].apply(lambda X: )

	hate_tweets = pd.DataFrame(hate_tweets['Text'].values.tolist())
	hate_tweets = pd.DataFrame(Scaler.fit_transform(hate_tweets))
	hate_tweets = pd.concat([scored_sentences, hate_tweets], axis=1)

	# print(hate_tweets)
	return hate_tweets, subtask_a_encode, subtask_b_encode, subtask_c_encode, words

hate_tweets, y1, y2, y3, vocab_len = load_data('dkhate.csv')



#x_train, x_test, y_train, y_test = split(hate_tweets, y1) #Subtask A #2 classes

#x_train, x_test, y_train, y_test = split(hate_tweets, y2) #Subtask B #4 classes

x_train, x_test, y_train, y_test = split(hate_tweets, y3) #Subtask C #3 classes

labels, counts = np.unique(y3, return_counts=True)
s_labels, s_counts = np.unique(y_test, return_counts=True)
print('Label-order and total count of entire dataset',labels, counts)
print('Label-order and count of testset',s_labels, s_counts)
# plt.bar(labels, counts, align='edge')
# plt.bar(s_labels, s_counts, align='center')
# plt.gca().set_xticks(labels)
# plt.show()

#simpler CNN:
# [[ 16  22  13   9]
#  [ 17  14   8   5]
#  [330 230 173 154]
#  [  5   6   5   3]]
# 0.13733333333333334

# [[ 35   8  14   3]
#  [ 31   4   7   2]
#  [533  82 209  63]
#  [ 10   1   8   0]]
# 0.26333333333333336

class Net(nn.Module):
	def __init__(self, input_size, padding_size, num_classes, fully_connected_layer):
		super(Net, self).__init__()
		self.conv1 = nn.Conv1d(input_size, fully_connected_layer, 2, padding=1)
		#self.conv2 = nn.Conv1d(50, fully_connected_layer , 2, padding=1)
		self.pool = nn.MaxPool1d(2, 2)
		self.fc1 = nn.Linear(fully_connected_layer, num_classes)
		# self.fc2 = nn.Linear(100, num_classes)
		self.dropout = nn.Dropout(0.1)

	def forward(self, x):
		# add sequence of convolutional and max pooling layers
		nsamples, nx,ny = x.shape
		x = x.reshape((nsamples,ny,nx))
		x = self.pool(nn.functional.relu(self.conv1(x)))
		#x = self.pool(nn.functional.relu(self.conv2(x)))
		x = x.view(x.shape[0], -1)
		#x = self.dropout(x)
		x = nn.functional.relu(self.fc1(x))
		#x = self.dropout(x)
		# x = nn.functional.relu(self.fc2(x))
		x = torch.sigmoid(x)
		#x = torch.softmax(x)	
		return x

#CNN Hyperparameters.
input_size = 70 #numr of features
fully_connected_layer = 25 #number of features in hidden state
padding_size = 2 #number of stacked lstm layers
num_classes = 4 #number of output classes 

#cnn1 = Net(input_size, padding_size, num_classes, fully_connected_layer)
#General Hyperparameters.

#task c LSTM

# [[ 10  43   7   0]
#  [  7  30   4   3]
#  [162 595 116  14]
#  [  1  15   3   0]]
# 0.20199999999999999

# [[  8   3  49   0]
#  [  3   4  37   0]
#  [ 63  67 749   8]
#  [  1   2  15   1]]
# 0.8346666666666667

# [[ 15   1  40   4]
#  [  8   0  27   9]
#  [197  15 586  89]
#  [  2   0  14   3]]
# 0.5333333333333333

# [[ 31   8  21   0]
#  [ 24   6  14   0]
#  [469 136 253  29]
#  [ 10   1   8   0]]
# 0.25866666666666666




class LSTM1(nn.Module):
	def __init__(self, num_classes, input_size, hidden_size, num_layers):
		super(LSTM1, self).__init__()
		self.num_classes = num_classes #number of classes
		self.num_layers = num_layers #number of layers
		self.input_size = input_size #input size
		self.hidden_size = hidden_size #hidden state

		self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
						  num_layers=num_layers, batch_first=True) #lstm
		self.fc_1 =  nn.Linear(hidden_size, num_classes) #fully connected 1
		#self.fc =    nn.Linear(128, num_classes) #fully connected last layer

		self.relu = nn.ReLU()
	
	def forward(self,x):
		h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
		c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
		# Propagate input through LSTM
		output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
		hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
		#out = self.relu(hn)
		out = self.fc_1(hn) #first Dense
		#out = self.relu(out) #relu
		#out = self.70(out) #Final Output
		#out = torch.sigmoid(out)
		#out = nn.functional.softmax(out)
		return out

# LSTM Hyperparameters
input_size =  70 #numb51er of features
hidden_size = 35 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers
num_classes = 4 #number of output classes 

lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers) #our lstm class

num_epochs = 100 #1000 epochs
learning_rate = 0.00001 #0.0001 for LSTM, 0.0000001 for CNN

#optim and loss
#criterion = torch.nn.BCELoss()    
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss() #if CBE, no softmax!
#optimizer = torch.optim.SGD(cnn1.parameters(), lr=learning_rate, momentum=0.9) 

batch_size = 500

train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

# train_loader = DataLoader((X_train, y_train), shuffle=True, batch_size=batch_size)
test_data = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_data, batch_size=batch_size)

lstm1.train()
#cnn1.train()

for epoch in range(num_epochs):
	for batch in train_loader:
		data, labels = batch

		# LSTM
		outputs = lstm1.forward(data) #forward pass
		optimizer.zero_grad() #caluclate the gradient, manually setting to 0
		#preds = torch.argmax(outputs, dim=1).float() #for BCE, not CEL
		#for CEL
		outputs = outputs.float()
		# print('labels',labels)
		preds = torch.argmax(outputs, dim=1).float()

		#CNN
		# outputs = cnn1.forward(data)
		# #print(outputs)
		# optimizer.zero_grad() #caluclate the gradient, manually setting to 0
		# preds = torch.argmax(outputs, dim=1).float()
		
		
		
		#print(preds)
		#ac = (preds == labels).numpy().mean() 
		#print('CNN AC: ', ac)
		loss = criterion(outputs, labels.long())
		loss = Variable(loss, requires_grad = True)
		loss.backward() #calculates the loss of the loss function
		optimizer.step() #improve from loss, i.e backprop
	if epoch % 10 == 0:
		#ac = (preds == labels).numpy().mean() 
		#print('CNN AC: ', ac)
		print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 

lstm1.eval()
#cnn1.eval()
val_accuracy = []
coll_preds = []
coll_target = []
with torch.no_grad():
	for data, target in test_loader:


		#LSTM
		output = lstm1(data)
		# loss = criterion(preds, target)
		# calculate the batch loss
		loss = criterion(output, target.long())
		loss = Variable(loss, requires_grad = True)
		loss.backward() #calculates the loss of the loss function
		optimizer.step() #improve from loss, i.e backprop
		preds = torch.argmax(output, dim=1).float()

		#CNN
		# output = cnn1(data)
		# print(target.dtype, output.dtype)
		# preds = torch.argmax(output, dim=1).float()
		# loss = criterion(output, target.long())
		# loss = Variable(loss, requires_grad = True)
		# loss.backward() #calculates the loss of the loss function
		# optimizer.step() #improve from loss, i.e backprop


		#general
		ac = (preds == target).numpy().mean() 
		print('CNN AC: ', ac)
		coll_preds += preds
		coll_target += target
		val_accuracy.append(ac)
print('F1score', f1_score(coll_preds, coll_target, average='micro'))
cnf_matrix = confusion_matrix(coll_target, coll_preds)
val_accuracy = np.mean(val_accuracy)
print(val_accuracy)
print(cnf_matrix)

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
	text = [word for word in text.split(' ')] # if word not in stopword] #stopwords
	text = " ".join(text)
	text = [lemmatizer.lemmatize('', word)[0] for word in text.split(' ')] #lemmatization
	text = " ".join(text) 
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
		if len(word_tokenize(data['Text'])) < 1 or len(word_tokenize(data['Text'])) > 30:
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
	print(suba.classes_)
	
	
	hate_tweets = hate_df['Text']


	
	
	hate_tweets = hate_tweets.apply(preprocess_text1)



	hate_tweets = pd.DataFrame(hate_tweets, columns = ['Text'])


	sentiments = hate_df['Text'].tolist()
	scored_sentences = [afinn.score(sent) for sent in sentiments]
	scored_sentences = pd.DataFrame(scored_sentences)
	
	
	
	#create label-dictionary, unnest lists, padding, scaling, concat
	tokenizer = Tokenizer()
	# MinMax = MinMaxScaler()
	Scaler = StandardScaler()

	#Count of data-len/vol
	input_counts = {}
	for index, data in hate_tweets.iterrows():
		in_len = len(word_tokenize(data['Text']))
		if in_len in input_counts:
			input_counts[in_len] += 1
		else:
			input_counts[in_len] = 1 

	#Plot/print data len/vol
	# print(dict(sorted(input_counts.items(), key=lambda item: item[1])))
	# print(dict(sorted(input_counts.items(), key=lambda item: item[0])))
	# print(hate_tweets["Text"].apply(lambda x: len(word_tokenize(x))).max())
	#plt.hist(hate_tweets['Text'].apply(lambda x: len(word_tokenize(x))), width = 20)
	plt.bar(list(input_counts.keys()), input_counts.values(), 10, color='g')
	plt.title("Histogram")  
	# Adding the legends
	plt.show()

	#Mechanism for dropping outliers for more similar data.
	

	# print(len(hate_tweets))



	tokenized = hate_tweets['Text'].apply(lambda X: tokenizer.fit_on_texts(X))
		
	hate_tweets['Text'] = hate_tweets['Text'].apply(lambda X: tokenizer.texts_to_sequences(X))

	hate_tweets['Text'] = hate_tweets['Text'].apply(lambda X: [float(item) for items in X for item in items]) #change for float?
	hate_tweets['Text'] = hate_tweets['Text'].apply(lambda X: pad_sequences([X], maxlen=35)) #change padlen
	hate_tweets['Text'] = hate_tweets['Text'].apply(lambda X: [float(item) for items in X for item in items])
	#hate_tweets['Text'] = hate_tweets['Text'].apply(lambda X: )

	hate_tweets = pd.DataFrame(hate_tweets['Text'].values.tolist())
	hate_tweets = pd.DataFrame(Scaler.fit_transform(hate_tweets))
	hate_tweets = pd.concat([scored_sentences, hate_tweets], axis=1)

	# print(hate_tweets)
	return hate_tweets, subtask_a_encode, subtask_b_encode, subtask_c_encode

hate_tweets, y1, y2, y3 = load_data('dkhate.csv')



x_train, x_test, y_train, y_test = split(hate_tweets, y1) #Subtask A #2 classes

#x_train, x_test, y_train, y_test = split(hate_tweets, y2) #Subtask B #4 classes

#x_train, x_test, y_train, y_test = split(hate_tweets, y3) #Subtask C #3 classes

labels, counts = np.unique(y1, return_counts=True)
s_labels, s_counts = np.unique(y_test, return_counts=True)
print('Label-order and total count of entire dataset',labels, counts)
print('Label-order and count of testset',s_labels, s_counts)

# plt.bar(labels, counts, align='edge')
# plt.bar(s_labels, s_counts, align='center')
# plt.gca().set_xticks(labels)
# plt.show()

class LSTM1(nn.Module):
	def __init__(self, num_classes, input_size, hidden_size, num_layers):
		super(LSTM1, self).__init__()
		self.num_classes = num_classes #number of classes
		self.num_layers = num_layers #number of layers
		self.input_size = input_size #input size
		self.hidden_size = hidden_size #hidden state

		self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
						  num_layers=num_layers, batch_first=True) #lstm
		self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
		self.fc =    nn.Linear(128, num_classes) #fully connected last layer

		self.relu = nn.ReLU()
	
	def forward(self,x):
		h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
		c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
		# Propagate input through LSTM
		output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
		hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
		out = self.relu(hn)
		out = self.fc_1(out) #first Dense
		#out = self.relu(out) #relu
		out = self.fc(out) #Final Output
		out = torch.sigmoid(out)
		#out = nn.functional.softmax(out)
		return out

# # LSTM Hyperparameters
input_size =  36#numb51er of features
hidden_size = 5 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers
num_classes = 2 #number of output classes 

lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers) #our lstm class


# class Net(nn.Module):
# 	def __init__(self, input_size, padding_size, num_classes, fully_connected_layer):
# 		super(Net, self).__init__()
# 		self.conv1 = nn.Conv1d(input_size, 200, 2, padding=2)
# 		self.conv2 = nn.Conv1d(200, 400 , 2, padding=1)
# 		self.conv3 = nn.Conv1d(400, fully_connected_layer, 2, padding=1)
# 		self.pool = nn.MaxPool1d(2, 2)
# 		self.fc1 = nn.Linear(fully_connected_layer, 100)
# 		self.fc2 = nn.Linear(100, num_classes)
# 		#self.dropout = nn.Dropout(0.1)

# 	def forward(self, x):
# 		# add sequence of convolutional and max pooling layers
# 		nsamples, nx,ny = x.shape
# 		x = x.reshape((nsamples,ny,nx))
# 		x = self.pool(nn.functional.relu(self.conv1(x)))
# 		x = self.pool(nn.functional.relu(self.conv2(x)))
# 		x = self.pool(nn.functional.relu(self.conv3(x)))
# 		x = x.view(x.shape[0], -1)
# 		#x = self.dropout(x)
# 		x = nn.functional.relu(self.fc1(x))
# 		#x = self.dropout(x)
# 		x = nn.functional.relu(self.fc2(x))
# 		x = torch.sigmoid(x)
# 		#x = torch.softmax(x)	
# 		return x

#simpler

# [[536   8  17 240]
#  [ 68   1   2  47]
#  [  5   0   1   1]
#  [ 55   1   1  27]] 54 % 
# [[310 101  54 336]
#  [ 47  13   7  51]
#  [  3   2   0   2]
#  [ 37   8   7  32]] # 0.3673333333333333


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
input_size = 36 #numr of features
fully_connected_layer = 15 #number of features in hidden state
padding_size = 2 #number of stacked lstm layers
num_classes = 2 #number of output classes 

#cnn1 = Net(input_size, padding_size, num_classes, fully_connected_layer)
#General Hyperparameters.

num_epochs = 100 #1000 epochs
learning_rate = 0.0000001

#optim and loss
criterion = torch.nn.BCELoss()    
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)

#criterion = nn.CrossEntropyLoss() #if CBE, no softmax!
#optimizer = torch.optim.SGD(cnn1.parameters(), lr=learning_rate, momentum=0.9) 

batch_size = 100

train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

# train_loader = DataLoader((X_train, y_train), shuffle=True, batch_size=batch_size)
test_data = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_data, batch_size=batch_size)

lstm1.train()
#cnn1.train()

# BCE CNN // LSTM 
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

		#print('argmax preds',preds)
		

		#CNN
		# outputs = cnn1.forward(data)
		# #print(outputs)
		# optimizer.zero_grad() #caluclate the gradient, manually setting to 0
		
		# labels = labels.float()
		# preds = torch.argmax(outputs, dim=1).float()
		
		# #print(labels.shape, labels.dtype, labels)
		
		#print(preds)
		#ac = (preds == labels).numpy().mean() 
		#print('CNN AC: ', ac)
		loss = criterion(preds, labels.float())
		loss = Variable(loss, requires_grad = True)
		loss.backward() #calculates the loss of the loss function
		optimizer.step() #improve from loss, i.e backprop
	if epoch % 10 == 0:
		ac = (preds == labels).numpy().mean() 
		#print('CNN AC: ', ac)
		print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 

lstm1.eval()
# cnn1.eval()
val_accuracy = []
coll_preds = []
coll_target = []
with torch.no_grad():
	for data, target in test_loader:


		# LSTM
		output = lstm1(data)
		#BCE
		preds = torch.argmax(output, dim=1).float()
		# loss = criterion(preds, target)
		# calculate the batch loss
		loss = criterion(preds, target.float())
		loss = Variable(loss, requires_grad = True)
		loss.backward() #calculates the loss of the loss function
		optimizer.step() #improve from loss, i.e backprop
		

		#CNN
		# output = cnn1(data)
		# preds = torch.argmax(output, dim=1).float()
		# loss = criterion(preds, target.float())
		# loss = Variable(loss, requires_grad = True)
		# loss.backward() #calculates the loss of the loss function
		# optimizer.step() #improve from loss, i.e backprop

		ac = (preds == target).numpy().mean() 
		#print('CNN AC: ', ac)
		val_accuracy.append(ac)
		coll_preds += preds
		coll_target += target
# 		
cnf_matrix = confusion_matrix(coll_target, coll_preds)
val_accuracy = np.mean(val_accuracy)
print('Overall Accuracy', val_accuracy)
print('F1score', f1_score(coll_preds, coll_target, average='binary'))
print(cnf_matrix)




#MULTICLASS PREDICTION


# for epoch in range(num_epochs):
# 	for batch in train_loader:
# 		data, labels = batch

# 		#LSTM
# 		# outputs = lstm1.forward(data) #forward pass
# 		# optimizer.zero_grad() #caluclate the gradient, manually setting to 0
# 		# #preds = torch.argmax(outputs, dim=1).float() #for BCE, not CEL
# 		# #for CEL
# 		# outputs = outputs.float()
# 		# # print('labels',labels)
# 		# preds = torch.argmax(outputs, dim=1).float()

# 		#print('argmax preds',preds)
		

# 		#CNN
# 		outputs = cnn1.forward(data)
	
# 		optimizer.zero_grad() #caluclate the gradient, manually setting to 0
		
		
# 		preds = torch.argmax(outputs, dim=1).float()
		
# 		#print(labels.shape, labels.dtype, labels)
		
# 		#print(preds)
# 		#ac = (preds == labels).numpy().mean() 
# 		#print('CNN AC: ', ac)
# 		loss = criterion(outputs, labels.long())
# 		loss = Variable(loss, requires_grad = True)
# 		loss.backward() #calculates the loss of the loss function
# 		optimizer.step() #improve from loss, i.e backprop
# 	if epoch % 10 == 0:
# 		#ac = (preds == labels).numpy().mean() 
# 		#print('CNN AC: ', ac)
# 		print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 
# print(preds)
# # lstm1.eval()
# cnn1.eval()
# val_accuracy = []
# coll_preds = []
# coll_target = []
# with torch.no_grad():
# 	for data, target in test_loader:


# 		#LSTM
# 		# output = lstm1(data)
# 		# #BCE
# 		# preds = torch.argmax(output, dim=1).float()
# 		# # loss = criterion(preds, target)
# 		# # calculate the batch loss
# 		# loss = criterion(preds, target.float())
# 		# loss = Variable(loss, requires_grad = True)
# 		# loss.backward() #calculates the loss of the loss function
# 		# optimizer.step() #improve from loss, i.e backprop
# 		# preds = torch.argmax(output, dim=1).float()
# 		# print(preds)


# 		#CNN
# 		output = cnn1(data)
# 		#print(target.dtype, output.dtype)
# 		preds = torch.argmax(output, dim=1).float()
# 		loss = criterion(output, target.long())
# 		loss = Variable(loss, requires_grad = True)
# 		loss.backward() #calculates the loss of the loss function
# 		optimizer.step() #improve from loss, i.e backprop
# 		ac = (preds == target).numpy().mean() 
# 		#print('CNN AC: ', ac)
# 		coll_preds += preds
# 		coll_target += target
# 		val_accuracy.append(ac)
# cnf_matrix = confusion_matrix(coll_target, coll_preds)
# print(cnf_matrix)
# val_accuracy = np.mean(val_accuracy)
# print(val_accuracy)
import pandas as pd
import numpy as np
import nltk
import re
#find danish stopwords

import string
import lemmy
import seaborn as sb

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing


from collections import Counter


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.metrics import f1_score

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.autograd import Variable

from afinn import Afinn
afinn = Afinn(emoticons=True)

import random

stopword = stopwords.words('danish')

lemmatizer = lemmy.load("da")


def preprocess_text ( text ):

	text = str(text).lower()
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
	text = [word for word in text.split(' ') if word not in stopword] #stopwords
	text = " ".join(text)
	text = [lemmatizer.lemmatize('', word)[0] for word in text.split(' ')] #lemmatization
	text = " ".join(text) 
	return text


def split(x, y):

	mm = MinMaxScaler()
	ss = StandardScaler()

	x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, shuffle=True)

	x_train = mm.fit_transform(x_train)
	x_test = mm.fit_transform(x_test)

	y_train= mm.fit_transform(y_train) 
	y_test= mm.fit_transform(y_test) 

	x_train_tensors = Variable(torch.Tensor(x_train))
	x_test_tensors = Variable(torch.Tensor(x_test))
	y_train_tensors = Variable(torch.Tensor(y_train))
	y_test_tensors = Variable(torch.Tensor(y_test)) 
	
	x_train_tensors_final = torch.reshape(x_train_tensors,   (x_train_tensors.shape[0], 1, x_train_tensors.shape[1]))
	x_test_tensors_final = torch.reshape(x_test_tensors,  (x_test_tensors.shape[0], 1, x_test_tensors.shape[1]))

	return x_train_tensors_final, x_test_tensors_final, y_train_tensors, y_test_tensors, x_train, x_test, y_train, y_test


def pad_input(sentences, seq_len):
	features = np.zeros((len(sentences), seq_len),dtype=int)
	for ii, review in enumerate(sentences):
		if len(review) != 0:

			features[ii, -len(review):] = np.array(review)[:seq_len]

	return features


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
	hate_df = pd.read_csv(tsv, delimiter="\t", header=[0])

	#labels
	hate_label = hate_df['subtask_a'].tolist()

	#make sentiment off of complete tweet
	hate_df['tweet'] = hate_df['tweet'].apply(preprocess_text1)
	sentiments = hate_df['tweet'].tolist()
	scored_sentences = [afinn.score(sent) for sent in sentiments]
	hate_df['sentiment'] = scored_sentences
	
	word_list = hate_df['tweet'].tolist()
	as_one = ''
	for sentence in word_list:
		as_one = as_one + ' ' + sentence
	words = as_one.split()

	counts = Counter(words)
	vocab = sorted(counts, key=counts.get, reverse=True)
	vocab_to_int = {word: ii for ii, word in enumerate(vocab, 0)}
	for i, sentence in enumerate(word_list):
		# Looking up the mapping dictionary and assigning the index to the respective words
		word_list[i] = [vocab_to_int[word] if word in vocab_to_int else 0 for word in sentence]

	#Padding to len=200
	seq_len = 200  # The length that the sentences will be padded/shortened to
	word_list = pad_input(word_list, seq_len)
	temp = pd.DataFrame(word_list)
	scored_sentences = pd.DataFrame(scored_sentences)

	#Y-encoding
	le = preprocessing.LabelEncoder()
	hate_label_encode = le.fit(hate_label)
	hate_label_encode = le.transform(hate_label)

	hate_label_encode = pd.DataFrame(data=hate_label_encode)

	hate_label_encode = hate_label_encode[:-1]
	temp = pd.concat([temp, scored_sentences], axis=1)
	temp = temp[:-1]
	# print(temp)
	return temp, hate_label_encode, counts


hate_df = load_data('dkhate/oe20da_data/offenseval-da-training-v1.tsv')

# X_train, X_test, y_train, y_test = split(hate_df[0], hate_df[1])

X_train, X_test, y_train, y_test, new_X_train, new_X_test, new_y_train, new_y_test = split(hate_df[0], hate_df[1])

vocab_index = hate_df[2]


#define data/set
batch_size = 64
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

# train_loader = DataLoader((X_train, y_train), shuffle=True, batch_size=batch_size)
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=batch_size)




# nsamples, nx = new_y_train.shape
# new_y_train = new_y_train.reshape((nsamples*nx))


# nsamples, nx = new_y_test.shape
# new_y_test = new_y_test.reshape((nsamples*nx))

# #clf = RandomForestClassifier(max_depth=2, random_state=42)
# clf = DecisionTreeClassifier(random_state=10)
# clf.fit(new_X_train, new_y_train)

# y_pred=clf.predict(new_X_test)
# print("Accuracy of CLF:", metrics.accuracy_score(new_y_test, y_pred))

class LSTM1(nn.Module):
	def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
		super(LSTM1, self).__init__()
		self.num_classes = num_classes #number of classes
		self.num_layers = num_layers #number of layers
		self.input_size = input_size #input size
		self.hidden_size = hidden_size #hidden state
		self.seq_length = seq_length #sequence length

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
		out = self.relu(out) #relu
		out = self.fc(out) #Final Output
		# out = torch.sigmoid(out)
		out = nn.functional.softmax(out)
		return out

num_epochs = 300 #1000 epochs
learning_rate = 0.00001 #0.001 lr

input_size = 201 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers
num_classes = 2 #number of output classes 

#Instantiate model
lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train.shape[1]) #our lstm class

#optim and loss
criterion = torch.nn.CrossEntropyLoss()    
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) 

lst1m.train()

for epoch in range(num_epochs):
	for batch in train_loader:
		data, labels = batch
		outputs = lstm1.forward(data) #forward pass
		optimizer.zero_grad() #caluclate the gradient, manually setting to 0
		#preds = torch.argmax(outputs, dim=1)

		loss = criterion(outputs, torch.max(labels, 1)[1])
		#print(loss)
		loss.backward() #calculates the loss of the loss function
		optimizer.step() #improve from loss, i.e backprop
	if epoch % 100 == 0:
		print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 

# # PATH = 'HateTorchModel'
# # torch.save(lstm1.state_dict(), PATH)

lstm1.eval()
# val_loss = 0
# val_accuracy = []
# val_correct = 0

# ac = 0
# for data, target in test_loader:
# 	with torch.no_grad():

# 		# move tensors to GPU if CUDA is available
		
# 		# forward pass: compute predicted outputs by passing inputs to the model
# 		output = lstm1(data)
# 		print(output)
# 		print(target)
# 		val_loss += criterion(output, torch.max(target, 1)[1])
# 		val_correct += (output.argmax(1) == target).type(torch.float).sum().item()

# # print(val_loss)
# val_correct = val_correct / len(test_loader.dataset)
# print(val_correct)


lstm_val_accuracy = []
with torch.no_grad():
	for data, target in test_loader:

		# move tensors to GPU if CUDA is available
		
		# forward pass: compute predicted outputs by passing inputs to the model
		target = target.float()
		output = lstm1(data)
		# calculate the batch loss
		loss = criterion(output, torch.max(target, 1)[1])
		# print(loss)
		# # update average validation loss 
		preds = torch.argmax(output, dim=1).flatten()
		#print(preds)
		lstm_ac = (preds == target).numpy().mean() * 100

		print('CNN AC: ', lstm_ac)

		lstm_val_accuracy.append(lstm_ac)

lstm_val_accuracy = np.mean(lstm_val_accuracy)
print(lstm_val_accuracy)





# 10 epochs
# ____________________________
# define the CNN architecture


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv1d(201, 100, 2, padding=1)
		self.conv2 = nn.Conv1d(100, 50, 2, padding=1)
		self.conv3 = nn.Conv1d(50, 25, 2, padding=1)
		self.pool = nn.MaxPool1d(2, 2)
		self.fc1 = nn.Linear(25, 10)
		self.fc2 = nn.Linear(10, 1)
		self.dropout = nn.Dropout(0.1)

	def forward(self, x):
		# add sequence of convolutional and max pooling layers
		nsamples, nx,ny = x.shape
		x = x.reshape((nsamples,ny,nx))
		x = self.pool(nn.functional.relu(self.conv1(x)))
		x = self.pool(nn.functional.relu(self.conv2(x)))
		x = self.pool(nn.functional.relu(self.conv3(x)))
		x = x.view(x.shape[0], -1)
		#x = self.dropout(x)
		x = nn.functional.relu(self.fc1(x))
		#x = self.dropout(x)
		x = nn.functional.relu(self.fc2(x))
		x = torch.sigmoid(x)	
		return x

model = Net()

learning_rate = 0.001 #0.001 lr
criterion = torch.nn.BCELoss()    
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

model.train()
train_loss = 0.0
accum = []

for epoch in range(1, 10):
	for data, target in train_loader:
		# move tensors to GPU if CUDA is available
		target = target.float()
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		# update training loss
		train_loss += loss.item()*data.size(0)
		# print(train_loss)

	print(((output.squeeze() > 0.5) == target.byte()).sum().item() / target.shape[0])

	print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 


model.eval()
val_loss = []
cnn_val_accuracy = []
with torch.no_grad():
	for data, target in test_loader:

		# move tensors to GPU if CUDA is available
		
		# forward pass: compute predicted outputs by passing inputs to the model
		target = target.float()
		output = model(data)
		# calculate the batch loss
		loss = criterion(output, target)
		# print(loss)
		# # update average validation loss 
		preds = torch.argmax(output, dim=1).flatten()
		#print(preds)
		cnn_ac = (preds == target).numpy().mean() * 100

		print('CNN AC: ', cnn_ac)

		cnn_val_accuracy.append(cnn_ac)

cnn_val_accuracy = np.mean(cnn_val_accuracy)
print(cnn_val_accuracy)






















# 		# valid_loss += loss.item()*data.size(0)

# 		# t = Variable(torch.FloatTensor([0.5]))  # threshold
# 		# out = (output > t).float() * 1
# 		# # print(out)

# 		# equals = target.float()  ==  out.t()
# 		# # print(equals)
# 		# #print(torch.sum(equals))
# 		# accuracy += (torch.sum(equals).numpy())
# 		# print(equals)
# 		# print(target)
# 	   # 

# # valid_loss = valid_loss/len(test_loader.dataset)
# # accuracy = accuracy/len(test_loader.dataset)

# # print(valid_loss)
# # print(accuracy)

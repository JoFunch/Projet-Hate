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
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


from collections import Counter


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.autograd import Variable

from afinn import Afinn
afinn = Afinn(emoticons=True)


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

	x_train = ss.fit_transform(x_train)
	x_test = ss.fit_transform(x_test)

	y_train= mm.fit_transform(y_train) 
	y_test= mm.fit_transform(y_test) 

	x_train_tensors = Variable(torch.Tensor(x_train))
	x_test_tensors = Variable(torch.Tensor(x_test))
	y_train_tensors = Variable(torch.Tensor(y_train))
	y_test_tensors = Variable(torch.Tensor(y_test)) 
	
	x_train_tensors_final = torch.reshape(x_train_tensors,   (x_train_tensors.shape[0], 1, x_train_tensors.shape[1]))
	x_test_tensors_final = torch.reshape(x_test_tensors,  (x_test_tensors.shape[0], 1, x_test_tensors.shape[1]))

	return x_train_tensors_final, x_test_tensors_final, y_train_tensors, y_test_tensors


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
	return temp, hate_label_encode, counts



hate_df = load_data('dkhate/oe20da_data/offenseval-da-training-v1.tsv')

X_train, X_test, y_train, y_test = split(hate_df[0], hate_df[1])

vocab_index = hate_df[2]


#define data/set
batch_size = 64
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

# train_loader = DataLoader((X_train, y_train), shuffle=True, batch_size=batch_size)
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=batch_size)

#define model

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
		self.fc = nn.Linear(128, num_classes) #fully connected last layer

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
		return out


num_epochs = 1000 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = 201 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers
num_classes = 1 #number of output classes 

#Instantiate model
lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train.shape[1]) #our lstm class

#optim and loss
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) 


for epoch in range(num_epochs):
	for batch in train_loader:
		data, labels = batch
		outputs = lstm1.forward(data) #forward pass
		optimizer.zero_grad() #caluclate the gradient, manually setting to 0
		  # obtain the loss function
		loss = criterion(outputs, labels)
		loss.backward() #calculates the loss of the loss function
		 
		optimizer.step() #improve from loss, i.e backprop
	if epoch % 100 == 0:
		print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 

PATH = 'HateTorchModel'
torch.save(lstm1.state_dict(), PATH)

lstm1.eval()

correct = 0
total = 0

with torch.no_grad():
	for imgs, labels in test_loader:
		batch_size = imgs.shape[0]
		outputs = lstm1(imgs)
		_, predicted = torch.max(outputs, dim=1)
		total += labels.shape[0]
		correct += int((predicted == labels).sum())
print("Accuracy: %f", correct / total)

#Accuracy: %f 54.956756756756754
# 10 epoch, 1 class, 

# Accuracy: %f 54.956756756756754
# 50 Epochs


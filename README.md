# Projet-Hate

Instructions for code:

Simply run data.py in > python 3.5. The script will both
	- initialise test-data, 
	- preprocess data,
	- instantiate model,
	- train,
	- save,
	- print test accuracy,
	- and predict an example, which may be modified towards end of the .py-file. 

By default, the script is set to run:
	num_epochs = 2000 #20 epochs
	learning_rate = 0.001 #0.001 lr
	input_size = 201 #number of features
	hidden_size = 2 #number of features in hidden state
	num_layers = 1 #number of stacked lstm layers
	num_classes = 1 #number of output classes 


Links:
- Data: https://figshare.com/articles/dataset/Danish_Hate_Speech_Abusive_Language_data/12220805 (download)
- Data: Description: https://sites.google.com/site/offensevalsharedtask/home
- Danish Sentiment Library: https://github.com/fnielsen/afinn
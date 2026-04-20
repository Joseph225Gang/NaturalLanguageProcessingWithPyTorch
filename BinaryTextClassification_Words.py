!pip install torchtext

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('datasets/ham-spam/spam.csv', encoding='latin-l')

data = data.drop(clumns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
data = data.rename(index = str, columns = {'v1': 'labels', 'v2': 'text'})

train, test = train_test_split(data, test_size = 0.2, random_state = 42)
train.reset_index(drop=True), test.reset_index(drop=True)

train.to_csv('datasets/ham-spam/train.csv', index=False)
test.to_csv('datasets/ham-spam/test.csv', index=False)

!ls datasets/ham-spam

import numpy as np

import torch
import torchtext

from torchtext.data import Field, BucketIterator, TabularDataset

import nltk
nltk.download('punkt')

from nltk import word_tokenize

TEXT = torchtext.data.Field(tokenize = word_tokenize)

LABEL = torchtext.data.LabelField(dtype = torch.float)
datafields = [("labels", LABEL), ("text", TEXT)]

trn, tst = torchtext.data.TabularDataset.splits(path = './datasets/ham-spam', train = 'train.csv', test = 'test.csv', format = 'csv', skip_header = True, fields = datafields)
TEXT.build_vocab(trn, max_size = 10500)
LABEL.build_vocab(trn)

batch_size = 64

train_iterator, test_iterator = torchtext.data.BucketIterator.splits(
	(trn, tst),
	batch_size = batch_size,
	sort_key = lambda x: len(x.text),
	sort_within_batch = False)

import torch.nn as nn

class RNN(nn.Module):
      def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
       super().__init__()
       self.embedding = nn.Embedding(input_dim, embedding_dim)
	   self.rnn = nn.LSTM(embedding_dim, hidden_dim)  @nn.RNN(embedding_dim, hidden_dim) 
       self.fc = nn.Linear(hidden_dim, output_dim)
       self.dropout = nn.Dropout(0.3)
      def forward(self, text):
       embedded = self.embedding(text)
	   embedded_dropout = self.dropout(embedded)
    output, (hidden, _) = self.rnn(embedded_dropout) @@output, hidden = self.rnn(embedded)
	   hidden_ID = hidden.squeeze(0)
       assert torch.equal(output[-1, :, :], hidden_ID)
       return self.fc(hidden_ID)

input_dim = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 1
model = RNN(input_dim, embedding_dim, hidden_dim, output_dim)
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr = le-6)
criterion = nn.BCEWithLogitsLoss()

def train(model, iterator, optimizer, criterion):
    
	epoch_loss = 0
	epoch_acc = 0

	model.train()

	for batch in iterator:

		optimizer.zero_grad()
		predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.labels)

		round_preds = torch.round(torch.sigmoid(predictions))
		correct = (rounded_preds == batch.labels).float()

		acc = correct.sum() / len(correct)

		loss.backward()

		optimizer.step()

		epoch_loss += loss.item()
		epoch_acc += acc.item()

		return epoch_loss / len(iterator, epoch_acc) / len(iterator)

num_epochs = 5
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

epoch_loss = 0
epoch_acc = 0
model.eval()

with torch.no_grad():
     for batch in test_iterator:
	 predictions = model(batch.text).squeeze(1)
	 loss = criterion(predictions, batch.labels)
	 rounded_preds = torch.round(torch.sigmoid(predictions))
	 correct = (rounded_preds == batch.labels).float()
         acc = correct.sum() / len(correct)

	 epoch_loss += loss.item()
         epoch_acc += acc.item()

test_loss = epoch_loss / len(test_iterator)
test_acc = epoch_acc / len(test_iterator)
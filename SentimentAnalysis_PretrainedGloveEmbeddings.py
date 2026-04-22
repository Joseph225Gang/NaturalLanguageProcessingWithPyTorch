import os
import spacy

import torch
import torchtext
from torchtext import datasets

import re
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

os.system("pip install spacy")
os.system("pyton -m spacy download en")

tweets = pd.read_csv('datasets/tweets/tweets.csv', on_bad_lines='skip')


tweets = tweets.drop(columns=['ItemID', 'SentimentSource'])
print(tweets.Sentiment.value_counts())

fig = plt.figure(figsize=(12,8))
ax = sns.barplot(x=tweets.Sentiment.unique(), y=tweets.Sentiment.value_counts())
ax.set(xlabel='Labels')

def tweet_clean(text):
    
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) 
    text = re.sub(r'https?:/\/\S+', ' ', text) 
    
    return text.strip()

from sklearn.model_selection import train_test_split

train, test = train_test_split(tweets, test_size=0.2, random_state=42)
train.reset_index(drop=True), test.reset_index(drop=True)
train.to_csv('datasets/tweets/train_tweets.csv', index=False)
test.to_csv('datasets/tweets/test_tweets.csv', index=False)

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def tokenizer(s):
    return [w.text.lower() for w in nlp(tweet_clean(s))]

TEXT = torchtext.data.Field(tokenize = tokenizer)
LABEL = torchtext.data.LabelField(dtype = torch.float)
datafields = [('Sentiment', LABEL),('SentimentText',TEXT)]
trn, tst = torchtext.data.TabularDataset.splits(path = 'datasets/tweets/',
                                                train = 'train_tweets.csv',
                                                test = 'test_tweets.csv',
                                                format = 'csv',
                                                skip_header = True,
                                                fields = datafields)
print(trn.examples[0])
TEXT.build_vocab(trn, max_size=25000,
                 vectors="glove.6B.100d",
                 unk_init=torch.Tensor.normal_)
LABEL.build_vocab(trn)

train_iterator, test_iterator = torchtext.data.BucketIterator.splits(
                                (trn, tst),
                                batch_size = 64,
                                sort_key=lambda x: len(x.SentimentText),
                                sort_within_batch=False)

import torch.nn as nn

class RNN(nn.Module):

      def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):

          super().__init__()
          self.embedding = nn.Embedding(vocab_size, embedding_dim)
          self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers,
                            bidirectional = bidirectional, dropout=dropout)
          self.fc = nn.Linear(hidden_dim*2, output_dim)
          self.dropout = nn.Dropout(dropout)

      def forward(self,text):
         embedded = self.dropout(self.embedding(text))
         output, hidden = self.rnn(embedded)
         hidden = self.dropout(torch.cat((hidden[-2,:,:],hidden[-1,:,:]),dim=1))
         return self.fc(hidden.squeeze(0))
      
input_dim = len(TEXT.vocab)
embedding_dim = 100

hidden_dim = 20
output_dim = 1

n_layers = 2
bidirectional = True

dropout = 0.5

model = RNN(input_dim,
            embedding_dim,
            hidden_dim,
            output_dim,
            n_layers,
            bidirectional,
            dropout)
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

unk_idx = TEXT.vocab.stoi[TEXT.unk_token]
pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
model.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)

import torch.optim as optim

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

def train(model, iterator, optimizer, criterion):

     epoch_loss = 0
     epoch_acc = 0

     model.train()

     for batch in iterator:

         optimizer.zero_grad()
         predictions = model(batch.SentimentText).squeeze(1)
         loss = criterion(predictions, batch.Sentiment)

         rounded_preds = torch.round(torch.sigmoid(predictions))
         correct = (rounded_preds == batch.Sentiment).float()

         acc = correct.sum() / len(correct)

         loss.backward()

         optimizer.step()

         epoch_loss += loss.item()
         epoch_acc += acc.item()
      
         return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
epoch_loss = 0
epoch_acc = 0    
model.eval()

with torch.no_grad():
 
     for batch in test_iterator:

         predictions = model(batch.SentimentText).squeeze(1)
         loss = criterion(predictions, batch.Sentiment)

         rounded_preds = torch.round(torch.sigmoid(predictions))
         correct = (rounded_preds == batch.Sentiment).float()

         acc = correct.sum()/len(correct)

         epoch_loss += loss.item()
         epoch_acc += acc.item()

test_loss = epoch_loss / len(test_iterator)
test_acc = epoch_acc / len(test_iterator)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

sentence = "That movie was really nice"
tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
indexed = [TEXT.vocab.stoi[t] for t in tokenized]
tensor = torch.LongTensor(indexed)
tensor = tensor.unsqueeze(1)
prediction = torch.sigmoid(model(tensor))
print(prediction.item())


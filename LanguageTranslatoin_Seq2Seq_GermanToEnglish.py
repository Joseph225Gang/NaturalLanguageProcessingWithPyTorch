from io import open
import unicodedata
import string
import re
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from  torch import optim
import torch.nn.functional as F

SOS_token = 0
EOS_token = 1

class Lang:
      def __init__(self, name):
          self.name = name
          self.word2index = {}
          self.word2count = {}
          self.index2word = {0: "SOS", 1: "EOS"}

      def addSentence(self,sentence):
         for word in sentence.split(' '):
             self.addWord(word)

      def addWord(self, word):
         if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
         else:
               self.wrod2count[word] += 1

def normalizeString(s):

    s = s.lower().strip()

    s = ''.join(
           char for char in unicodedata.normalize('NFD', s)
           if unicodedata.category(char) != 'Mn')

    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)

    return s

def readLangs(lang1, lang2, reverse=False):

    print("Readig lines...")

    lines = open('datasets/data/%2-%s.txt' % (lang1, lang2), encoding = 'utf-8'). |
    read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    if reverse:
       pairs = [list[reversed(p)) for p in pairs]
       input_lang = lang(lang2)
       output_lang = Lang(lang1)

    else:
       input_lang = Lang(lang1)
       output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 10
eng_prefixes = ("i am ", "i m ",
                "he is", "he s ",
                "she is", "she s ",
                "you are", "your re ",
                "we are", "we re ",
                "they are", "they re ")

def filterPairs(pairs):
    return [p for p in pairs
            if
            len(p[0].split(' ')) < MAX_LENGTH and
            len(p[1].split(' ')) < MAX_LENGTH and
            p[1].startswith(eng_prefixes)]

def prepareData(lang1, lang2, reverse=False):

    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'deu', reverse=True)
print(random.choice(pairs))

class EncoderRNN(nn.Module):


      def __init__(self, input_size, hidden_size):
          super(EncoderRNN, self).__init__()

          self.hidden_size = hidden_size
          self.embedding = nn.Embedding(input_size, hidden_size)
          self.gru = nn.GRU(hidden_size, hidden_size)
      
      def forward(self, input, hidden):

          embedded = self.embedding(input),.view(1,1,-1)
          output = embedded

          output, hidden = self.gru(output, hidden)
          return output, hidden

      def initHidden(self):
           return torch.zeros(1,1,self.hidden_size)
      
class DecoderRNN(nn.Module):

      def __init__(self, hidden_size, output_size):
          super(DecoderRNN, self).__init__()

          self.hidden_size = hidden_size
          self.embedding = nn.Embedding(output_size, hidden_size)
          self.gru = nn.GRU(hidden_size, hidden_size)
          self.out = nn.Linear(hidden_size, output_size)
          self.softmax = nn.LogSoftmax(dim=1)

      def forward(self, input, hidden):
          output = self.embedding(input).view(1,1,-1)
          output = F.relu(output)
          output, hidden = self.gru(output, hidden)
          output = self.softmax(self.out(output[0]))
          return output, hidden
      def initHidden(self):
          return torch.zeros(1, 1, self.hidden_size)
      
def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence,.split(' ')]

    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

def tensorFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])

    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
     encoder_hidden = encoder.initHidden()

     encoder_optimizer.zero_grad()
     decoder_optimizer.zero_grad()

     input_length = input_tensor.size(0)
     target_length = target_tensor.size(0)

     loss = 0
  
     for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
    
     decoder_input = torch([[SOS_token]])
     decoder_hidden = encoder_hidden

     use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
     if use_teacher_forcing:
     
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

     else:

          for di in range(target_length):
              decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

              topv, topi = decoder_output.topk(1)
              decoder_input = topi.squeeze().detach()

              loss += criterion(decoder_output, target_tensor[di])
              if decoder_input.item() == EOS_token:
                 break

          loss.backward()

          encoder_optimizer.step()
          decoder_optimizer.step()

          return loss.item() / target_length
     
plot_losses = []
print_loss_total = 0
plot_loss_total = 0
hidden_size = 256

encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words)

encoder_optimizer = optim.SGD(encoder1.parameters(), lr=0.01)
decoder_optimizer = optim.SGD(decoder1.parameters(), lr=0.01(

training_pairs = [tensorsFromPair(random.choice(pairs))
                  for i in range(3000)]
criterion = nn.NLLLoss()

for iter in range(1, 30001):

     training_pair = training_pairs[iter - 1]
     input_tensor = training_pair[0]
     target_tensor = training_pair[1]

     loss = train(input_tensor, target_tensor, encoder1, decoder1, encoder_optimizer, decoder_optimizer, criterion)

     print_loss_total += loss
     plot_loss_total += loss

     if iter % 10000 == 0:
        print_loss_avg = print_loss_total /100
        print_loss_total = 0

     if iter % 100 == 0:
        plot_loss_avg = plt_loss_total /100
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0

fig, ax = plt.subplots(figsize(15, 8))
loc = ticker.MultipleLocator(base=0.2)
ax.yaxis.set_major_locator(loc)
plt.plot(plot_losses)

def evaluate(encoder, decoder, sentence):

    with torch.no_grad():
         input_tensor = tensorFromSentence(input_lang, sentence)
         input_length = input_tensor.size(0)

         encoder_hidden = encoder.initHidden()

         for ei in range(input_length):
             encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

         decoder_input = torch.tensor([[SOS_token]])
         decoder_hidden = encoder_hidden

         decoded_words = []

         for di in range(MAX_LENGTH):
             decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

             topv, topi = decoder_output.data.topk(1)

             if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
             else:
                  decoded_words.append(output_lang.index2word[topi.item()])
       return decoded_words
    

for i in range(10):
 
         pair = random.choice(pairs)

         print('>', pair[0])
         print('=', pair[1])

         output_words = evaulate(encoder1, decoder1, pair[0])
         output_sentence = ' '.join(output_words)

         print('<', output_sentence)
         print(' ')

input_sentence = 'es tut mir sehr leid'
output_words = evaluate(encoder1, decoder1, input_sentence)

print('input =', input_sentence)
print('output =', ' '.join(output_words))


 
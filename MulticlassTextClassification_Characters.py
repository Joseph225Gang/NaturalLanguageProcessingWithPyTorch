from io import open
import glob
import os

print(glob.glob('datasets/data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + "   .,;'"
n_letters = len(all_letters)

language_names = {}

all_languages = []

def unicodeToAscii(s):
    return ''.join(
	c for c in unicodedata.normalize('NFD', s)
	if unicodedata.category(c) != 'Mn'
        and c in all_letters
  )


import torch
def letterToTensor(letter):

     tensor = torch.zeros(1, n_letters)
     tensor[0][all_letters.find(letter)] = 1

     return tensor
print(letterToTensor('a'))

def nameToTensor(name):
    tensor = torch.zeros(len(name), 1, n_letters)

    for li, letter in enumerate(name):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor
def findFiles(path): 
    return glob.glob(path)


mary_tensor = nameToTensor('Mary')

mary_tensor.size()

import torch.nn as nn
total_names = 0

for filename in findFiles('datasets/data/names/*.txt'):
    
    language = os.path.splitext(os.path.basename(filename))[0]
    
    all_languages.append(language)
    
    read_names = open(filename, encoding='utf-8').read().strip().split('\n')
    
    names = [unicodeToAscii(line) for line in read_names]
    
    language_names[language] = names
    
    total_names += len(names)


n_languages = len(all_languages)

class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        
        hidden = self.i2h(combined)
        
        output = self.i2o(combined)
        output = self.softmax(output)
        
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 256
hidden = torch.zeros(1, n_hidden)
rnn = RNN(n_letters, n_hidden, n_languages)
inp = letterToTensor('C')
hidden, next_hidden = rnn(inp, hidden)

inp = nameToTensor('Charron')
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(inp[0], hidden)

def languageFromOutput(output):
    # 檢查 output 是否包含元素
    if output.numel() == 0 or output.size(-1) == 0:
        return "Unknown", -1
    
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_languages[category_i], category_i

print(languageFromOutput(output))
n_languages = len(all_languages)

import random


def randomTrainingExample():
    
    random_language_index = random.randint(0, n_languages - 1)
    language = all_languages[random_language_index]
    
    random_language_names = language_names[language]
    
    name = random_language_names[random.randint(0, len(random_language_names) - 1)]
    
    language_tensor = torch.tensor([all_languages.index(language)], dtype=torch.long)
    name_tensor = nameToTensor(name)
    
    return language, name, language_tensor, name_tensor

for i in range(10):
    language, name, language_tensor, name_tensor = randomTrainingExample()
    
    print('language =', language, ', name =', name)

criterion = nn.NLLLoss()
learning_rate = 0.005 

def train(langauge_tensor, name_tensor):
    
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(name_tensor.size()[0]):
        output, hidden = rnn(name_tensor[i], hidden)

    loss = criterion(output, langauge_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

n_iters = 200000

current_loss = 0
all_losses = []

for epoch in range(1, n_iters + 1):
    
    language, name, language_tensor, name_tensor = randomTrainingExample()
    
    output, loss = train(language_tensor, name_tensor)
    current_loss += loss

    if epoch % 5000 == 0:
        guess, guess_i = languageFromOutput(output)
        correct = '✓' if guess == language else '✗ (%s)' % language
        
        print('%d %d%% %.4f %s / %s %s' % (epoch, 
                                           epoch / n_iters * 100,
                                           loss,
                                           name, 
                                           guess, 
                                           correct))

    if epoch % 1000 == 0:
        all_losses.append(current_loss / 1000)
        current_loss = 0

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
plt.plot(all_losses)
plt.show()

n_predictions = 3
input_name = 'Batsakis'

with torch.no_grad():
    
    name_tensor = nameToTensor(input_name)
    
    hidden = rnn.initHidden()
    for i in range(name_tensor.size()[0]):
        output, hidden = rnn(name_tensor[i], hidden)

    topv, topi = output.topk(n_predictions, 1, True)

    for i in range(n_predictions):
        
        value = topv[0][i].item()
        language_index = topi[0][i].item()
        
        print('(%.2f) %s' % (value, all_languages[language_index]))
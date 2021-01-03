import numpy as np
import pandas as pd
import nltk
import re
from tqdm import tqdm
import pickle
import gc
from random import sample
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

######################### Loading the text corpus ###########################
# Text file containing sentences
f = open('corpus/text.txt', 'r')

# Number of lines(sentences)
num_lines = sum(1 for line in open('corpus/text.txt'))

# Loading the vocab of the corpus where each word is mapped to a unique id
with open('corpus/word2id.pickle', 'rb') as handle:
    word2id = pickle.load(handle)

# Add an end symbol to the dictonary
word2id['$end$'] = len(word2id)

############################### Here comes Deep learning ##########################################

############# Parameters ############
epochs = 100
lr = 1e-4
embedding_size = 150
hidden_size = 1000
vocab_size = len(word2id)

save_path = 'trained_models/'

############# Model Definition ############
class LSTM_lm(nn.Module):
    def __init__(self, nIn, nHidden, vocab):
        super(LSTM_lm, self).__init__()
        self.nIn = nIn
        self.embeddings = nn.Embedding(vocab, nIn)
        self.lstm = nn.LSTM(nIn, nHidden, bidirectional = True)
        self.lin = nn.Linear(2*nHidden, vocab)

    def forward(self, sentence):
        sentence = self.embeddings(sentence).view(-1, 1, self.nIn)
        recurrent, (hidden, c) = self.lstm(sentence)
        hidden = hidden.view(1, -1)
        out = self.lin(recurrent)
        s,b,o = out.size()
        out = out.view(s*b,o)
        return out


# Using gpu if available else cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = LSTM_lm(embedding_size, hidden_size, vocab_size)
model = model.to(device)

############# Loss Function ############
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


############# Training!!!! ############
for epoch in range(epochs):
    total_loss = 0

    # f will reach to end after epoch, so load it again
    f = open('corpus/text.txt', 'r')

    for s in tqdm(f, total = num_lines):
        s = s.split()
        s = s[::-1]
        s.append('$end$')
        s = [word2id[i] for i in s]

        inp = s[:-1]
        target = s[1:]
        # Input and target tensor
        inp = torch.tensor(inp).to(device)
        target = torch.tensor(target).to(device)
        # Getting model output
        out = model(inp)

        # Backpropagation
        optimizer.zero_grad()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        total_loss += loss

    avg_loss = total_loss/num_lines

    print('Epoch: ' + str(epoch + 1) + '/' + str(epochs) + ' Loss: ' + str(avg_loss))

    print('Saving model...')
    path = save_path + type(model).__name__ + '_FinalLoss_' + str(avg_loss) + '.pt'
    torch.save(model.state_dict(), path)
    print('Model saved at the path ' + path)

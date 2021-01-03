import numpy as np
import pandas as pd
import nltk
import re
from tqdm import tqdm
import pickle
import gc
from random import sample
import pdb
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pronouncing

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if os.path.isfile('pretrained_embeds/glove.6B/glove.6B.50d_w2v.txt'):
    glove_model = KeyedVectors.load_word2vec_format("pretrained_embeds/glove.6B/glove.6B.50d_w2v.txt", binary=False)
else:
    glove2word2vec(glove_input_file="pretrained_embeds/glove.twitter.27B.25d.txt", word2vec_output_file="pretrained_embeds/gensim_glove_vectors.txt")
    glove_model = KeyedVectors.load_word2vec_format("pretrained_embeds/gensim_glove_vectors.txt", binary=False)

def get_embed(word):
    # Case folding
    word = word.lower()
    try:
        return (glove_model.get_vector(word))
    except:
        return (glove_model.get_vector('<unk>'))


# Loading the vocab of the corpus where each word is mapped to a unique id(which was used for training)
with open('corpus/word2id.pickle', 'rb') as handle:
    word2id = pickle.load(handle)
# Add an end symbol to the dictonary
word2id['$end$'] = len(word2id)

# Invert the word2id dictnoary
id2word = {v: k for k, v in word2id.items()}

# Same parameters used for training
embedding_size = 100
hidden_size = 1200
vocab_size = len(word2id)

save_path = 'trained_models/'

# Defining the model in same way as training
class LSTM_lm(nn.Module):
    def __init__(self, nIn, nHidden, vocab):
        super(LSTM_lm, self).__init__()
        self.nIn = nIn
        self.embeddings = nn.Embedding(vocab, nIn)
        self.lstm = nn.LSTM(nIn, nHidden)
        self.lin = nn.Linear(nHidden, vocab)

    def forward(self, sentence):
        sentence = self.embeddings(sentence).view(-1, 1, self.nIn)
        recurrent, (hidden, c) = self.lstm(sentence)
        hidden = hidden.view(1, -1)
        out = self.lin(recurrent)
        s,b,o = out.size()
        out = out.view(s*b,o)
        return out

# Using gpu if available else cpu
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

model = LSTM_lm(embedding_size, hidden_size, vocab_size)
model = model.to(device)

trained_model = '1.pt'

if device != 'cpu':
	model.load_state_dict(torch.load(save_path + trained_model))
else:
	model.load_state_dict(torch.load(save_path + trained_model, map_location='cpu'))


# Geneartes sentence given the last word
def generate_sentence(word):
	next_word_id = word2id[word]
	inp = []
	out_sent = [word]

	for i in range(10):
		inp.append(next_word_id)
		inp_tensor = torch.tensor(inp).to(device)

		out = model(inp_tensor)
		out = out.cpu()

		next_word_id = torch.max(out, 1)[1].numpy()[-1]
		next_word_id = int(next_word_id)

		if next_word_id == len(word2id)-1:
			break

		out_sent.append(id2word[next_word_id])

	out_sent = out_sent[::-1]
	out_sent = ' '.join(out_sent)
	return out_sent



def rap_gen(input_word):
	# Get rhyme word
	rhyms = pronouncing.rhymes(input_word)
	scores = {}
	for w in rhyms:
	    try:
	        scores[w] = glove_model.similarity(w, input_word)
	    except:
	        pass
	scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

	rwords = [i[0] for i in scores]
	rwords = rwords[:20]

	generated_sentences = []

	for i in rwords:
		try:
			line = generate_sentence(i)
			if len(line.split()) > 1:
				generated_sentences.append(line)
		except:
			pass

	generated_sentences = [i for i in generated_sentences if 'i put it' not in i]

	first_line = generate_sentence(input_word)
	if 'i put it' in first_line:
		generated_sentences = generated_sentences[:4]
	else:
		generated_sentences = [first_line] + generated_sentences[:3]
		# print(first_line)

	return('\n'.join(generated_sentences))




if __name__ == "__main__":
	while(1):
		input_word = input('enter:')
		print(rap_gen(input_word))
	


# 	# pdb.set_trace()


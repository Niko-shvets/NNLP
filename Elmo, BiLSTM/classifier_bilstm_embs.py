import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter, defaultdict
import codecs
import matplotlib.pyplot as plt
import re
import seaborn as sns
import string
from time import time
import string 


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as data_utils


translator = str.maketrans('', '', string.punctuation)

def surround_non_symbols(word):
    new_word=''
    list_letters=list(word)    
    for symbol in list_letters:
        if symbol in set(string.punctuation):
            symbol=' '+symbol+' '
        else:
            symbol=symbol
        new_word+=symbol
    return new_word
    

def preprocess_text(Text,punct=False,figures=False):
    result=[]
    for sentense in Text:
        string=(sentense.lower())
        string = " ".join([surround_non_symbols(word) for word in string.split()])
        clear_sentence=" ".join(string.split())
        if punct==True:
            clear_sentence=clear_sentence.translate(translator)
        if figures==True:
            clear_sentence=re.sub(r'\d+', '', clear_sentence)
        result.append(clear_sentence)
    return result

def tokenization(data):
    data_tok =[line.split() for line in tqdm(data)]
    return data_tok

def vocab_creator(data):
  vocab=set()
  for sentence in data:
    vocab.update(sentence)
  return vocab

def load_embeddings(emb_path, vocab):
    clf_embeddings = {}
    emb_vocab = set()
    for line in open(emb_path):
        line = line.strip('\n').split()
        word, emb = line[0], line[1:]
        emb = [float(e) for e in emb]
        if word in vocab:
            clf_embeddings[word] = emb
    for w in vocab:
        if w in clf_embeddings:
            emb_vocab.add(w)
    word2idx = {w: idx for (idx, w) in enumerate(emb_vocab)}
    max_val = max(word2idx.values())
    
    word2idx['UNK'] = max_val + 1
    word2idx['EOS'] = max_val + 2
    emb_dim = len(list(clf_embeddings.values())[0])
    clf_embeddings['UNK'] = [0.0 for i in range(emb_dim)]
    clf_embeddings['EOS'] = [0.0 for i in range(emb_dim)]
    
    embeddings = [[] for i in range(len(word2idx))]
    for w in word2idx:
        embeddings[word2idx[w]] = clf_embeddings[w]
    embeddings = torch.Tensor(embeddings)
    return embeddings, word2idx

def to_matrix(lines, vocab, max_len=None, dtype='int32'):
    """Casts a list of lines into a matrix"""
    pad = vocab['EOS']
    max_len = max_len or max(map(len, lines))
    lines_ix = np.zeros([len(lines), max_len], dtype) + pad
    for i in range(len(lines)):
        line_ix = [vocab.get(l, vocab['UNK']) for l in lines[i]]
        lines_ix[i, :len(line_ix)] = line_ix
    lines_ix = torch.LongTensor(lines_ix)
    return lines_ix

def reorganize_labels(labels):
  lab=[]
  for ly in labels:
    if ly==0:
      lab.append([1,0])
    else:
      lab.append([0,1])
  return lab

def data_train(tokens:list,labels:list,vocab:dict):
  data=[]
  for idx, (t, l) in enumerate(zip(tokens, labels)):
    t = to_matrix([t], vocab)
    l = torch.Tensor([l])
    data.append((t,l))
  return data
def data_test(tokens:list,vocab:dict):
  data=[]
  for idx, t in enumerate(tokens):
    t = to_matrix([t], vocab)
    
    data.append(t)
  return data  

def binary_accuracy(preds, y):
    # y is either [0, 1] or [1, 0]
    # get the class (0 or 1)
    y = torch.argmax(y, dim=1)
    
    # get the predicted class
    preds = torch.argmax(torch.sigmoid(preds), dim=1)
    
    correct = (preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

class BiLSTM(nn.Module):
    def __init__(self, embeddings, hidden_dim=128, lstm_layer=1, output=2):
        
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        # load pre-trained embeddings
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        # embeddings are not fine-tuned
        self.embedding.weight.requires_grad = False
        
        # RNN layer with LSTM cells
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer, 
                            bidirectional=True)
        # dense layer


        self.output = nn.Linear(hidden_dim*2, output)
    
    def forward(self, sents):
        x = self.embedding(sents)
        
        # the original dimensions of torch LSTM's output are: (seq_len, batch, num_directions * hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # reshape to get the tensor of dimensions (seq_len, batch, num_directions, hidden_size)
        lstm_out = lstm_out.view(x.shape[0], -1, 2, self.hidden_dim)#.squeeze(1)
        
        # lstm_out[:, :, 0, :] -- output of the forward LSTM
        # lstm_out[:, :, 1, :] -- output of the backward LSTM
        # we take the last hidden state of the forward LSTM and the first hidden state of the backward LSTM
        dense_input = torch.cat((lstm_out[-1, :, 0, :], lstm_out[0, :, 1, :]), dim=1)
        
        #hidden = self.linear(dense_input)
        y=self.output(dense_input).view([1, 2])
 
        return y

def train_epoch(model, train_data, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    # set the model to the training mode
    model.train(mode=True)
    
    for t, l in train_data:
        # reshape the data to n_words x batch_size (here batch_size=1)
        t = t.view((-1, 1))
        # transfer the data to GPU to make it accessible for the model and the loss
        t = t.to(device)
        l = l.to(device)
        
        # set all gradients to zero
        optimizer.zero_grad()
        
        # forward pass of training
        # compute predictions with current parameters
        predictions = model(t)
        # compute the loss
        loss = criterion(predictions, l)
        # compute the accuracy (this is only for report)
        acc = binary_accuracy(predictions, l)
        
        # backward pass (fully handled by pytorch)
        loss.backward()
        # update all parameters according to their gradients
        optimizer.step()
        
        # data for report
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(train_data), epoch_acc / len(train_data)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# device = torch.device('cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(texts,labels):
  train_labels= np.array([1 if l == 'pos' else 0 for l in labels ])
  preprocessed_train = preprocess_text(texts,True,False)   
  train_tokens=tokenization(preprocessed_train)
  voc=vocab_creator(train_tokens)
  embeddings, vocab = load_embeddings('glove.6B.300d.txt', voc)


  train_labels=reorganize_labels(train_labels)
  train_data=data_train(train_tokens,train_labels,vocab)
  hidden_dim = 128
  layers = 1

  model = BiLSTM(embeddings, hidden_dim, lstm_layer=layers)
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.BCEWithLogitsLoss()

  
  model = model.to(device)
  #model.load_state_dict(torch.load('tut1-model.pt', map_location=device))
  criterion = criterion.to(device)
  # train_accs=[]
  # valid_accs=[]
  for epoch in tqdm(range(3)):
    start_time = time()
    
    train_loss, train_acc = train_epoch(model, train_data, optimizer, criterion)
    print('train accuracy on  ',epoch,' epoch ',train_acc,' loss ',train_loss)

  return [model,vocab]

def classify(text,params):
  
  preprocessed_train = preprocess_text(text,True,False)   
  train_tokens=tokenization(preprocessed_train)
  vocab=params[1]
  train_data=data_test(train_tokens,vocab)
  model=params[0]
  model.eval()
  predicts=[]

  with torch.no_grad():
    for t in train_data:
      t = t.view((-1, 1))
      t = t.to(device)
      predictions = model(t)
      pred=torch.argmax(torch.sigmoid(predictions), dim=1)
      predicts.append(int(pred.detach().cpu().numpy()))
  y_predict=np.array(['pos' if l == 1 else 'neg' for l in predicts ])
  return y_predict
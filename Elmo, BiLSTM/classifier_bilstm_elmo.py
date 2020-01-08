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
from torch.utils.data import Dataset, DataLoader
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

ELMO_OPTIONS = "elmo_2x2048_256_2048cnn_1xhighway_options.json"
ELMO_WEIGHT = "elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
from allennlp.modules.elmo import Elmo, batch_to_ids


class elmo_LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128, lstm_layer=1, output=1):
        
        super(elmo_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        # self.embedding_dim=embedding_dim
        # self.batch_size=batch_size
        self.lstm_layer=lstm_layer
        # self.embedding = elmo

        
        # RNN layer with LSTM cells
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer, 
                            bidirectional=True)
        # dense layer
        self.output = nn.Linear(hidden_dim*2, hidden_dim)
        self.output1=nn.Linear(hidden_dim, output)

    
    def forward(self, x):
        
        # x = self.embedding(sents)["elmo_representations"][0]
        lstm_out, _ = self.lstm(x)
        

        lstm_out = lstm_out.view(x.shape[1], -1, 2, self.hidden_dim)
        

        # dense_input = torch.cat((lstm_out[-1, :, 0, :], lstm_out[0, :, 1, :]), dim=1)
        dense_input = torch.cat((lstm_out[:,-1,0,:], lstm_out[:,0,1,:]), dim=1)

        y=self.output(dense_input)
        y=self.output1(y)
        return y
    
def train(text,labels):
  elmo = Elmo(ELMO_OPTIONS, ELMO_WEIGHT, num_output_representations = 1).cuda()
  hidden_dim = 128
  layers = 1
  model = elmo_LSTM(embedding_dim=512, hidden_dim=hidden_dim,lstm_layer=layers)
  optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=4e-4)
  criterion = nn.BCEWithLogitsLoss()
  device = torch.device('cuda')
  model = model.to(device)
  #model.load_state_dict(torch.load('tut1-model.pt', map_location=device))
  criterion = criterion.to(device)
  preprocessed_train = preprocess_text(text,True,False) 

  train_tokens=tokenization(preprocessed_train)
  tr_lab=np.array([1 if l == 'pos' else 0 for l in labels ])
  data_set=[[train_tokens[i], torch.Tensor([tr_lab[i]])] for i in range(len(train_tokens))]
  train_loader=DataLoader(data_set,batch_size=128)
  sigmoid=nn.Sigmoid()
  accuracy=[]
  losses=[]
  for epoch in range(50):
    epoch_accuracy=[]
    epoch_loss=[]
    model.train(True)
    for x,y in train_loader:
      optimizer.zero_grad()
      x = batch_to_ids(x).cuda()
      X=elmo(x)['elmo_representations'][0]
      y=y.cuda()
      predictions=model(X)
      y_pred=sigmoid(predictions).detach().cpu().numpy().round()
      loss = criterion(predictions, y)
      acc = accuracy_score(y_pred,y.detach().cpu().numpy())
      loss.backward()
      optimizer.step()

      epoch_accuracy.append(acc)
      epoch_loss.append(loss.detach().cpu().numpy())
    model.train(False)
    accuracy.append(np.mean(epoch_accuracy))
    losses.append(np.mean(epoch_loss))
    print(accuracy[-1])
  return model

def classify(text,model):
  elmo = Elmo(ELMO_OPTIONS, ELMO_WEIGHT, num_output_representations = 1).cuda()
  preprocessed_train = preprocess_text(text,True,False) 
  train_tokens=tokenization(preprocessed_train)
  data_set=[train_tokens[i] for i in range(len(train_tokens))]
  train_loader=DataLoader(data_set,batch_size=128)
  sigmoid=nn.Sigmoid()
  y_preds=[]
  model.eval()
  for x in train_loader:
    x = batch_to_ids(x).cuda()
    X=elmo(x)['elmo_representations'][0]
    preds=model(X)
    y_pred=sigmoid(preds).detach().cpu().numpy().round()
    y_preds.append(y_pred)
  y_preds=np.concatenate(y_preds)
  y_predict=np.array(['pos' if l == 1 else 'neg' for l in y_preds ])
  return y_preds
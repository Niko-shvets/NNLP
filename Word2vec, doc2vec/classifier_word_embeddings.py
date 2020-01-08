import numpy as np

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
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
    data_tok =[line.split() for line in data]
    return data_tok

def sent_vec(sent,glove_model):
    wv_res = np.zeros(glove_model.vector_size)
    ctr = 1
    for w in sent:
        if w in glove_model:
            ctr += 1
            wv_res += glove_model[w]
    wv_res = wv_res/ctr
    #return (wv_res, ctr)
    return wv_res

def vector(tokens:list,glove_model):
  doc_vecs=[]
  for sentence in (tokens):
    doc_vecs.append(sent_vec(sentence,glove_model))
  return doc_vecs
def tensor_from_array(array_list:list):
  tensors=[torch.FloatTensor(array_list[ind]) for ind in range(len(array_list))]
  return tensors

def dataLoader(X,y,batchsize):
  y=torch.from_numpy(np.array(y)).float()
  tensor_set=torch.stack([torch.Tensor(i) for i in X])
  data_set=data_utils.TensorDataset(tensor_set,y)
  train_loader=data_utils.DataLoader(data_set,batch_size=batchsize,shuffle=True)
  return train_loader


class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.tanh = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 10)
            self.tanh1 = torch.nn.ReLU()
            self.fc3=torch.nn.Linear(10,1)
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            hidden = self.fc1(x)
            tanh = self.tanh(hidden)

            output = self.fc2(tanh)
            output = self.tanh1(output)

            output = self.fc3(output)
            output = self.sigmoid(output)
            return output
        
def train(text,labels):
  batch_size=50  
  preprocessed_train = preprocess_text(text,True,False)
  train_tokens=tokenization(preprocessed_train)

  labels = np.array([1 if l == 'pos' else 0 for l in labels ])
  glove_input_file = 'glove.6B.300d.txt'
  word2vec_output_file = 'glove.6B.300d.w2vformat.txt'
  glove2word2vec(glove_input_file, word2vec_output_file)
  glove_model_300 = KeyedVectors.load_word2vec_format("glove.6B.300d.w2vformat.txt", binary=False)

  train_doc_vecs=vector(train_tokens,glove_model_300)
  train_tensors=tensor_from_array(train_doc_vecs) 

  train_loader=dataLoader(train_tensors,labels,batch_size)  
  model = Feedforward(300, 50)#.cuda()
  optimizer = optim.Adam(model.parameters())
  loss_function = nn.BCEWithLogitsLoss()
  loader=dataLoader(train_tensors,labels,batch_size)
  train_losses=[]
  train_accuracy=[]
  for epoch in range((500)):
    model.train(True)
    loss_epoch=[]
    accuracy_epoch=[]
    for x_batch,y_batch in (loader):

      prediction=model(x_batch)

      loss=loss_function(prediction.view(-1),y_batch)

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      loss_epoch.append(loss.data.cpu().numpy())

      accuracy_epoch.append(accuracy_score(prediction.detach().cpu().numpy().round(),y_batch.detach().cpu().numpy()))

    model.train(False)
    train_losses.append(np.mean(loss_epoch))
    train_accuracy.append(np.mean(accuracy_epoch))
    if epoch%10==0:
        print('epoch: ',epoch,' accuracy: ',train_accuracy[-1],' loss: ', train_losses[-1])
  return model

def classify(text,model):
  preprocessed_train = preprocess_text(text,True,False)
  train_tokens=tokenization(preprocessed_train)

  glove_input_file = 'glove.6B.300d.txt'
  word2vec_output_file = 'glove.6B.300d.w2vformat.txt'
  glove2word2vec(glove_input_file, word2vec_output_file)
  glove_model_300 = KeyedVectors.load_word2vec_format("glove.6B.300d.w2vformat.txt", binary=False)

  train_doc_vecs=vector(train_tokens,glove_model_300)
  train_tensors=tensor_from_array(train_doc_vecs) 

  preds=[]
  model.eval()
  tensor_set=torch.stack([torch.Tensor(i) for i in train_tensors])
  data_set=data_utils.TensorDataset(tensor_set,torch.zeros(tensor_set.shape[0]),)
  loader=data_utils.DataLoader(data_set,batch_size=50,shuffle=False)
  
  for test_batch,y in loader:
    predictions = model(test_batch)
    preds.append(predictions.detach().numpy().round())
  y_pred=np.concatenate(preds).reshape(-1)
  y_predict=np.array(['pos' if l == 1 else 'neg' for l in y_pred ]) 
    
  return y_predict

import numpy as np

from sklearn.metrics import accuracy_score
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter, defaultdict
import codecs
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

class LogisticRegressionBow(nn.Module):
  def __init__(self, vectors,num_labels):
    super().__init__()
    self.linear = nn.Linear(vectors, num_labels)
    #self.sigmoid=torch.nn.Sigmoid()
  def forward(self, bow_vec):
    linear=self.linear(bow_vec)
   # res=self.sigmoid(linear)
    return linear



def train(text,labels):
    batch_size=50
    preprocessed_train = preprocess_text(text,True,False)
    train_tokens=tokenization(preprocessed_train)
    
    labels = np.array([1 if l == 'pos' else 0 for l in labels ])
    # glove_input_file = 'glove.6B.300d.txt'
    # word2vec_output_file = 'glove.6B.300d.w2vformat.txt'
    # glove2word2vec(glove_input_file, word2vec_output_file)
    fasttext = KeyedVectors.load_word2vec_format("wiki-news-300d-1M.vec", binary=False)
    
    train_doc_vecs=vector(train_tokens,fasttext)
    train_tensors=tensor_from_array(train_doc_vecs) 
    
    train_loader=dataLoader(train_tensors,labels,batch_size)
    model = LogisticRegressionBow(300, 2)#.cuda()
    optimizer = optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()
    for epoch in (range(50)):
     # model.train(True)

        for X,y in (train_loader):
      
            y=y.type(torch.LongTensor)
            optimizer.zero_grad()
            l_probs = model(X)

            loss = loss_function(l_probs, y)
            loss.backward()
            optimizer.step()
            

    return model

def classify(data,model):
  batchsize=50 
  preprocessed_train = preprocess_text(data,True,False)
  train_tokens=tokenization(preprocessed_train)
  fasttext = KeyedVectors.load_word2vec_format("wiki-news-300d-1M.vec", binary=False)

  train_doc_vecs=vector(train_tokens,fasttext)
  train_tensors=tensor_from_array(train_doc_vecs)  

  preds=[]
  model.eval()
  tensor_set=torch.stack([torch.Tensor(i) for i in train_tensors])
  data_set=data_utils.TensorDataset(tensor_set,torch.zeros(tensor_set.shape[0]),)
  loader=data_utils.DataLoader(data_set,batch_size=batchsize,shuffle=False)
  
  for test_batch,y in loader:
    predictions = model(test_batch)
    preds.append(np.argmax(predictions.detach().numpy(),axis=1))
  y_pred=np.concatenate(preds)
  y_predict=np.array(['pos' if l == 1 else 'neg' for l in y_pred ])
  return y_predict


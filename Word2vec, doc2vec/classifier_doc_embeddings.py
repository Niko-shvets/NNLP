import numpy as np
from sklearn.metrics import accuracy_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter, defaultdict
import codecs
import numpy as np
from sklearn.metrics import accuracy_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
from sklearn.linear_model import LogisticRegression
import pandas as pd
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

model = Doc2Vec(vector_size=300, min_count=1, epochs=55)
def gensim_preprocess_tain(data,labels):
  train_documents = [TaggedDocument(gensim.utils.simple_preprocess(doc), [label]) for  doc,label in zip(data,labels)]
  return train_documents

def gensim_preprocess_test(data):
  preprocessed=[gensim.utils.simple_preprocess(sentence) for sentence in data]
  return preprocessed

def vectors_creature(model,tokens_list:list):
  vectors=[model.infer_vector(sentence) for sentence in (tokens_list) ]
  return vectors

def train(text,labels):
    print(labels[:2])
    labels = np.array([1 if l == 'pos' else 0 for l in labels ])
    train_gensim_data=gensim_preprocess_tain(text,labels)
    data_train=gensim_preprocess_test(text)
    model.build_vocab(train_gensim_data)
    model.epochs=2
    model.train(train_gensim_data, total_examples=model.corpus_count, epochs=model.epochs) 
    vectors=vectors_creature(model,data_train)
    logreg = LogisticRegression(C=0.01,penalty='l2',max_iter=200,random_state=42,solver='saga')
    logreg.fit(vectors,labels)
    param=logreg
    return param

def classify(data,param):
    data_train=gensim_preprocess_test(data)
    vectors=vectors_creature(model,data_train)
    y_pred=param.predict(vectors) 
    y_predict=np.array(['pos' if l == 1 else 'neg' for l in y_pred ])
    return y_predict



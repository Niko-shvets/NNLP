from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class LR:
    def __init__(self,lr,num_iter,alpha):
        self.lr=lr
        self.num_iter=num_iter
        self.alpha=alpha
        self.accuracy_list=[]
        self.losses=[]
        self.accuracy_list_dev=[]
        self.losses_dev=[]
        
    def sigmoid(self, z): 
        return 1./(1+np.exp(-z))
    
    def loss(self,y,h):
        return -(y*np.log(h+1e-8)+(1-y)*np.log(1-h+1e-8)).mean() + (self.alpha*self.w**2).sum()

    def grad(self,X,y,h):
        self.gradient = X.T.dot(h-y)/y.size + 2*self.alpha*self.w

    def fit(self, X, y,X_dev,y_dev):

        #X_coo = hstack([np.ones((X.shape[0],1)), X])
       # X_coo_dev = hstack([np.ones((X_dev.shape[0],1)), X_dev])
        self.w=np.zeros(X.shape[1])
        #print(self.w.shape)

        for i in tqdm(range(self.num_iter)):
            #trainig
            z = X.dot(self.w)
            h = self.sigmoid(z)
            self.grad(X, h, y)
            self.w+=self.lr*self.gradient
            #loss and accuracy for train
            loss=self.loss(y,h)
            if i%100==0:
                accuracy=accuracy_score(y,h.round())
                self.accuracy_list.append(accuracy)
                self.losses.append(loss)
                
            #prediction for dev
                z1=X_dev.dot(self.w)
                h1=self.sigmoid(z1)
            #loss and accuracy for dev
                self.losses_dev.append(self.loss(y_dev,h1))
                self.accuracy_list_dev.append(accuracy_score(y_dev,h1.round()))

    def predict(self,X):
        #X_=hstack([np.ones((X.shape[0],1)),X])
        print(X.shape)
        z=X.dot(self.w)
        prediction=self.sigmoid(z)
        return prediction.round()
            
from collections import Counter, defaultdict
import numpy as np

def N_grams(tokens, ngram_range):
    return [" ".join(tokens[idx:idx+i]) 
            for i in range(ngram_range[0],ngram_range[1]+1) 
            for idx in range(len(tokens)-i+1)]

class BernoulliNB(object):
    def __init__(self,positive_tokens,negative_tokens):
        self.positive_tokens=positive_tokens
        self.negative_tokens=negative_tokens

    def fit(self, positive_voc, negative_voc):
        pos_probs = defaultdict(lambda: 1/(len(self.positive_tokens)+1*2))
        neg_probs = defaultdict(lambda: 1/(len(self.negative_tokens)+1*2))
        
        for i,j in positive_voc.items():
            pos_probs[i]=j/(len(self.positive_tokens)+1*2)
            
        for i,j in negative_voc.items():
            neg_probs[i]=j/(len(self.negative_tokens)+1*2)
        
        self.unique_words = sorted(list(set(pos_probs.keys())|set(neg_probs.keys())))

        self.positive_probs={w: pos_probs[w] for w in self.unique_words}
        self.negative_probs={w: neg_probs[w] for w in self.unique_words}
        self.log_pos_prior=np.log(len(self.positive_tokens) /(len(self.positive_tokens)+len(self.negative_tokens)))
        self.log_neg_prior=np.log(len(self.negative_tokens) /(len(self.positive_tokens)+len(self.negative_tokens)))

        
    def predict(self, X,N):
        prediction=[]
        for sentence in (X):
            predictions_positive=self.log_pos_prior
            predictions_negative=self.log_neg_prior
            for word in N_grams(sentence,N):
                if word in self.positive_probs:
                    predictions_positive=predictions_positive-np.log(1-self.positive_probs[word])+np.log(self.positive_probs[word])
                if word in self.negative_probs:
                    predictions_negative=predictions_negative-np.log(1-self.negative_probs[word])+np.log(self.negative_probs[word])
            predictions_positive+=self.log_pos_prior
            predictions_negative+=self.log_neg_prior
            prediction.append(int(predictions_positive>predictions_negative))
        return prediction
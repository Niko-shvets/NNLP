import numpy as np
from collections import Counter, defaultdict

def N_grams(tokens, ngram_range):
    return [" ".join(tokens[idx:idx+i]) 
            for i in range(ngram_range[0],ngram_range[1]+1) 
            for idx in range(len(tokens)-i+1)]

class multinomial():
    def __init__(self,positive_tokens,negative_tokens):
        self.positive_tokens=positive_tokens
        self.negative_tokens=negative_tokens
        
    def fit(self,positive_voc,negative_voc):
        unique_words=sorted(list(set(positive_voc.keys())|set(negative_voc.keys())))
        
        self.all_positive=sum(positive_voc.values())
        self.all_negative=sum(negative_voc.values())
        
        self.positive_probs={w:positive_voc[w] for w in unique_words}
        self.negative_probs={w:negative_voc[w] for w in unique_words}
        
        self.log_pos_prior=np.log(len(self.positive_tokens) /(len(self.positive_tokens)+len(self.negative_tokens)))
        self.log_neg_prior=np.log(len(self.negative_tokens) /(len(self.positive_tokens)+len(self.negative_tokens)))
        
    def predict(self, X,N):
        prediction=[]
        for sentence in X:
            predictions_positive=self.log_pos_prior
            predictions_negative=self.log_neg_prior
            for word in N_grams(sentence,N):
                if word in self.positive_probs:
                    predictions_positive+=np.log(self.positive_probs[word])-np.log(self.all_positive)
                    
                if word in self.negative_probs:
                    predictions_negative+=np.log(self.negative_probs[word])-np.log(self.all_positive)
            
            predictions_positive+=self.log_pos_prior
            predictions_negative+=self.log_neg_prior
            prediction.append(int(predictions_positive>predictions_negative))
        return prediction
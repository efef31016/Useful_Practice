import numpy as np
from math import pi

class Gaussian_NB():
    def __init__(self, data):

        '''
        data[:,:-1] : features
        data[:,-1] : class
        '''
        self.data = data
        
    def fit(self):
        self.y = self.data[:,-1]
        # mean and s.d.
        self.class_dict = {i:data[data[:,-1]==i,:-1] for i in np.unique(self.y)}
        # prior probability
        self.class_prob = {i:np.mean(data[:,-1]==i) for i in np.unique(self.y)}
        summaries = dict()
        for class_value, rows in self.class_dict.items():
            m = np.mean(rows,axis = 0)
            s = np.std(rows,axis = 0)
            summaries[class_value] = np.vstack([m,s])
        self.summaries = summaries
        
    def joint_log_likelihood(self, test):
        log_likelihood = []
        for i in np.unique(self.y):
            log_class_prob = np.log(self.class_prob[i])
            log_class_likelihood = -0.5*np.log(2*pi*np.square(self.summaries[i][1]))
            log_class_likelihood = log_class_likelihood-0.5*(test-self.summaries[i][0])**2/np.square(self.summaries[i][1])
            log_likelihood.append(log_class_prob+np.sum(log_class_likelihood, axis = 1))
        return np.array(log_likelihood).T
    
    def predict(self,test):
        log_likelihood = self.joint_log_likelihood(test)
        pre = np.argmax(log_likelihood, axis = 1)
        return pre
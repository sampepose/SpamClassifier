# -*- coding: utf-8 -*-

import numpy as np
from scipy.misc import logsumexp

class naive_bayes:
    
    def __init__(self, Method='gaussian'):
        self.Method = Method
        
    # X: M x N
    # y: M x 1
    def fit(self, X, y):
        unq, unq_counts = np.unique(y, return_counts=True)
        self.priors = unq_counts / np.sum(unq_counts) # K x 1
        self.num_classes = len(unq)

        mean = []
        var = []
        for y_i in unq:
            X_i = X[y == y_i, :]

            mean.append(np.mean(X_i, axis=0))
            var.append(np.var(X_i, axis=0))
        
        self.mean = np.vstack(mean) # K x N
        self.var = np.vstack(var)  # K x N

    def _joint_log_likelihood(self, X):
        prob = []

        epsilon = 1e-9        
        
        for k in range(self.num_classes):
            mean = self.mean[k, :]
            var = self.var[k, :] + epsilon # add epsilon so we never divide by zero
            gauss = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            gauss -= 0.5 * np.sum(np.square(X - mean) / var, axis=1)
            prob.append(np.log(self.priors[k]) + gauss)

        prob = np.vstack(prob).T
        return prob
    
    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        log_prob = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob).T
    
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
        
    def score(self, X, y):
        pred = self.predict(X)
        
        score = 0.0
        for i in range(pred.shape[0]):
            if (pred[i] == y[i]):
                score += 1
        
        return score / pred.shape[0]
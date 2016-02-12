# -*- coding: utf-8 -*-

import numpy as np
from scipy.misc import logsumexp

# Base class inspired by https://github.com/scikit-learn/scikit-learn/blob/a1860144aa2083277ba354b0cc46f9eb4acf0db0/sklearn/naive_bayes.py
class NaiveBayes:

    def fit(self, X, y):
        """Fit the Naive Bayes model to the input

        Arguments:
        X -- M x N numpy array
        y --  M x 1 numpy array, storing K unique labels

        Returns:
        None
        """

        raise NotImplementedError()

    def _predict_log_proba(self, X):
        """Predict the log of the label probabilities for the input

        Arguments:
        X -- M x N numpy array

        Returns:
        log_probabilities -- M x K numpy array
        """

        jll = self._joint_log_likelihood(X)
        log_prob = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob).T

    def predict_proba(self, X):
        """Predict the label probabilities for the input

        Arguments:
        X -- M x N numpy array

        Returns:
        probabilities -- M x K numpy array
        """

        return np.exp(self._predict_log_proba(X))

    def predict(self, X):
        """Predict the labels for the input

        Arguments:
        X -- M x N numpy array

        Returns:
        probabilities -- M x K numpy array
        """

        return self._classes[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def score(self, X, y):
        """Accuracy for test data and expected labels

        Arguments:
        X -- M x N numpy array
        y --  M x 1 numpy array, storing K unique labels

        Returns:
        accuracy_score -- decimal value (0.0-1.0)
        """

        pred = self.predict(X)

        score = 0.0
        for i in range(pred.shape[0]):
            if (pred[i] == y[i]):
                score += 1

        return score / pred.shape[0]

class GaussianBayes(NaiveBayes):

    def fit(self, X, y):
        """Fit the Naive Bayes model to the input

        Arguments:
        X -- M x N numpy array
        y --  M x 1 numpy array, storing K unique labels

        Returns:
        None
        """

        unq, unq_counts = np.unique(y, return_counts=True)

        self._classes = unq # K x 1
        self.priors = unq_counts / y.shape[0] # K x 1
        self.num_classes = len(unq)

        mean = []
        var = []

        for y_i in unq:
            X_i = X[y == y_i, :]

            mean.append(np.mean(X_i, axis=0))
            var.append(np.var(X_i, axis=0))

        self.mean = self._weights = np.vstack(mean) # K x N
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

class MultinomialBayes(NaiveBayes):

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        """Fit the Naive Bayes model to the input

        Arguments:
        X -- M x N numpy array
        y --  M x 1 numpy array, storing K unique labels

        Returns:
        None
        """

        unq, unq_counts = np.unique(y, return_counts=True)

        self._classes = unq # K x 1
        self._log_priors = np.log(unq_counts) - np.log(y.shape[0])

        # Alpha will be used for smoothing later.
        # If set to zero, we could have numerical instability.
        if self.alpha == 0.0:
            self.alpha = 1e-16

        feature_log_probs = []
        for k in range(len(unq)):
            # Grab all data for the kth label
            subset = X[y == k, :]

            # We add alpha for smoothing. This means we don't take the
            # log of zero in case a feature is missing (=> P(feature) = 0)
            counts = np.sum(subset, axis=0) + self.alpha
            count_sum = np.sum(counts) + self.alpha * 2

            # Subtract the logs (same as division)
            feature_log_probs.append(np.log(counts) - np.log(count_sum.reshape(-1,1)))

        self._feature_log_prob = np.vstack(feature_log_probs)

    def _joint_log_likelihood(self, X):
        """Predict the log of the label probabilities for the input

        Arguments:
        X -- M x N numpy array

        Returns:
        log_probabilities -- M x K numpy array
        """

        # Multinomial Bayes is a simple linear classifier in log-space!
        return self._log_priors + X.dot(self._feature_log_prob.T)

class BernoulliBayes(NaiveBayes):

    def __init__(self, alpha=1.0, binarize=0.5):
        self.alpha = alpha
        self.binarize = binarize

    def __binarize(self, X):
        X_bin = np.zeros(X.shape)
        X_bin[X > self.binarize] = 1
        return X_bin

    def fit(self, X, y):
        """Fit the Naive Bayes model to the input

        Arguments:
        X -- M x N numpy array
        y --  M x 1 numpy array, storing K unique labels

        Returns:
        None
        """

        unq, unq_counts = np.unique(y, return_counts=True)

        self._classes = unq # K x 1
        self._priors = unq_counts / y.shape[0] # K x 1

        if self.binarize is not None:
            X = self.__binarize(X)

        # Alpha will be used for smoothing later.
        # If set to zero, we could have numerical instability.
        if self.alpha == 0.0:
            self.alpha = 1e-16

        feature_log_probs = []
        for k in range(len(unq)):
            # Grab all data for the kth label
            subset = X[y == k, :]

            # We add alpha for smoothing. This means we don't take the
            # log of zero in case a feature is missing (=> P(feature) = 0)
            counts = np.sum(subset, axis=0) + self.alpha
            count_sum = np.sum(counts) + self.alpha * 2

            # Subtract the logs (same as division)
            feature_log_probs.append(np.log(counts) - np.log(count_sum.reshape(-1,1)))

        self._feature_log_prob = np.vstack(feature_log_probs)

    def _joint_log_likelihood(self, X):
        """Predict the log of the label probabilities for the input

        Arguments:
        X -- M x N numpy array

        Returns:
        log_probabilities -- M x K numpy array
        """

        if self.binarize is not None:
            X = self.__binarize(X)

        # log of the Bernoulli equation
        neg_prob = np.log(1. - np.exp(self._feature_log_prob))
        log_priors = np.log(self._priors)
        return X.dot((self._feature_log_prob - neg_prob).T) + neg_prob.sum(axis=1) + log_priors

# -*- coding: utf-8 -*-

from load_data import load_data
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import operator

word_labels = ['address', 'all', '3d', 'our', 'over', 'remove', 'internet',
               'order', 'mail', 'receive', 'will', 'people', 'report', 'addresses',
               'free', 'business', 'email', 'you', 'credit', 'your', 'font', '000',
               'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs', 'telnet', '857',
               'data', '415', '85', 'technology', '1999', 'parts', 'pm', 'direct',
               'cs', 'meeting', 'original', 'project', 're', 'edu', 'table', 'conference']

class METHOD:
    gaussian, multinomial, bernoulli = range(3)

method = METHOD.bernoulli
iterations = 50
k = 5
binarize = True

def find_hyperparams_bernoulli(clf, X, y):
    # Set the parameters by cross-validation
    param_grid = [{'binarize': [x * 10**-2 for x in range(0, 5000)]}]
    grid = GridSearchCV(clf, param_grid)
    grid.fit(X, y)
    print('done fitting')
    return grid.best_estimator_

def show_auc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def top_k_features(k, weights):
    return sorted(zip(word_labels, weights), reverse=True, key=operator.itemgetter(1))[:k]

def binarize(X, thresh):
    X_bin = np.zeros(X.shape)
    X_bin[X > thresh] = 1
    return X_bin

scores = []
roc_auc = []
weights = []

for i in range(iterations):
    X_train, X_test, y_train, y_test = load_data(Train=True)

    # For now, let's train only on word frequency vectors
    X_train = X_train[:, 0:48]
    X_test = X_test[:, 0:48]

    if method == METHOD.gaussian:
        # Gaussian Naive Bayes
        # This doesn't really make sense here because our features aren't continuous
        # in a way that is Gaussian, they are percentages. This might be better
        # for things like number of capital letters.
        clf = GaussianNB()

    if method == METHOD.multinomial:
        # Multinomial Naive Bayes
        clf = MultinomialNB(alpha=1.0)

    if method == METHOD.bernoulli:
        # Bernoulli (multi-variate) Naive Bayes
        # It doesn't make sense to include features that are inherently differentiated by magnitude,
        # i.e total number of capital letters. So we should only test on word frequencies.
        clf = BernoulliNB(alpha=1.0, binarize=0.31) # binarize found via cross validation
    #    X, y = load_data()
    #    print(find_hyperparams_bernoulli(clf, X[:, 0:48], y))

    if binarize:
        X_train = binarize(X_train, 0.31)
        X_test = binarize(X_test, 0.31)

    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))

    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    roc_auc.append(auc(fpr, tpr))

    if method == METHOD.gaussian:
        weights = clf.theta_
    else:
        weights = clf.feature_log_prob_

show_auc(y_test, clf.predict_proba(X_test)[:, 1])

print('Accuracy. Avg: %0.5f, Std: %0.5f' % (np.mean(scores), np.std(scores)))
print('AUC. Avg: %0.5f, Std: %0.5f' % (np.mean(roc_auc), np.std(roc_auc)))
print('Top %d features:' % k)
print(clf.classes_)
print(top_k_features(k, weights[0, :]))
print(top_k_features(k, weights[1, :]))

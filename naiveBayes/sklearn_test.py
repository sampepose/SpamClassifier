# -*- coding: utf-8 -*-

from load_data import load_data
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.grid_search import GridSearchCV

class METHOD:
    gaussian, multinomial, bernoulli = range(3)
    
method = METHOD.bernoulli
iterations = 50
    
def find_hyperparams_bernoulli(clf, X, y):
    # Set the parameters by cross-validation
    param_grid = [{'binarize': [x * 10**-2 for x in range(0, 5000)]}]
    grid = GridSearchCV(clf, param_grid)
    grid.fit(X, y)
    print('done fitting')
    return grid.best_estimator_

avg = 0

for i in range(iterations):
    X_train, X_test, y_train, y_test = load_data(Train=True)

    # For now, let's train only on word frequency vectors
    X_train = X_train[:, 0:48]
    X_test = X_test[:, 0:48]

    if method == METHOD.gaussian:
        # Gaussian Naive Bayes
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
        
    clf.fit(X_train, y_train)
    avg += clf.score(X_test, y_test)
    
print(avg / iterations)
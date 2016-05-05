from load_data import load_data
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
from sklearn.metrics import roc_curve, auc
import pydot_ng
import operator

word_labels = ["make", "address", "all", "3d", "our", "over", "remove", "internet",
                "order", "mail", "receive", "will", "people", "report", "addresses",
                "free", "business", "email", "you", "credit", "your", "font", "000",
                "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
                "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
                "meeting", "original", "project", "re", "edu", "table", "conference"]

iterations = 50
k = 5

def find_hyperparams(clf, X, y):
    # Set the parameters by cross-validation
    param_grid = [{'n_estimators': [x for x in range(50, 100)],
                    'criterion': ['gini', 'entropy'],
                    'max_features': ['sqrt', 'log2', None ]}]
    grid = GridSearchCV(clf, param_grid, n_jobs=-1)
    grid.fit(X, y)
    print('done fitting')
    return grid.best_estimator_

def top_k_features(k, weights):
    return sorted(zip(word_labels, weights), reverse=True, key=operator.itemgetter(1))[:k]

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

scores = []
roc_auc = []

for i in range(iterations):
    X_train, X_test, y_train, y_test = load_data(Train=True)

    # For now, let's train only on word frequency vectors
    X_train = X_train[:, 0:48]
    X_test = X_test[:, 0:48]

    clf = RandomForestClassifier(criterion="gini", max_features="log2", n_estimators=73)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))

    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    roc_auc.append(auc(fpr, tpr))



#X, y = load_data()
#print(find_hyperparams(clf, X, y))

show_auc(y_test, clf.predict_proba(X_test)[:, 1])
print(top_k_features(5, clf.feature_importances_))

#dot_data = StringIO()
#export_graphviz(clf, out_file=dot_data, class_names=["Ham", "Spam"], feature_names=word_labels)
#graph = pydot_ng.graph_from_dot_data(dot_data.getvalue())
#graph.write_png("tree.png")

print('Accuracy. Avg: %0.5f, Std: %0.5f' % (np.mean(scores), np.std(scores)))
print('AUC. Avg: %0.5f, Std: %0.5f' % (np.mean(roc_auc), np.std(roc_auc)))

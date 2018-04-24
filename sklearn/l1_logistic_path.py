from sklearn import linear_model
from sklearn.svm import l1_min_c
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.datasets import make_classification
import pandas as pd

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# a = pd.read_csv('sample20170117_labeled_0207.csv')
# X = a.values[0: 100, 0: 110]
# y = a.values[0: 100, 110]
# y = np.array([1 if i == 1. else -1 for i in y])

X, y = make_classification(n_samples=1000, n_features=100, n_classes=2)

cs = l1_min_c(X, y, loss='log') * np.logspace(0, 3)
clf = linear_model.LogisticRegression(penalty='l1')
coefs_ = []
for c in cs:
    clf.set_params(C=c)
    clf.fit(X, y)
    coefs_.append(clf.coef_.ravel().copy())

coefs_ = np.array(coefs_)
plt.plot(np.log10(cs), coefs_)
ymin, ymax = plt.ylim()
plt.xlabel('log(c)')
plt.ylabel('coefficients')
plt.title('logistic regression path')
plt.axis('tight')
plt.show()

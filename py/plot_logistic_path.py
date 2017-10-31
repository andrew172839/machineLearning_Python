"""
Path with L1- Logistic Regression
Computes path on IRIS dataset.
"""

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import datasets
from sklearn.svm import l1_min_c

#iris = datasets.load_iris()
#X = iris.data
#y = iris.target

#X = X[y != 2]
#y = y[y != 2]

import pandas as pd
a = pd.read_csv('sample20170117_labeled_0207.csv')
X = a.values[0: 100, 0: 110]
y = a.values[0: 100, 110]
y = np.array([1 if i == 1. else -1 for i in y])

X -= np.mean(X, 0)

cs = l1_min_c(X, y, loss='log') * np.logspace(0, 3)

print("Computing regularization path ...")
start = datetime.now()
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
coefs_ = []
for c in cs:
    clf.set_params(C = c)
    clf.fit(X, y)
    coefs_.append(clf.coef_.ravel().copy())
print("This took ", datetime.now() - start)

coefs_ = np.array(coefs_)
plt.plot(np.log10(cs), coefs_)
ymin, ymax = plt.ylim()
plt.xlabel('log(C)')
plt.ylabel('Coefficients')
plt.title('Logistic Regression Path')
plt.axis('tight')
plt.show()
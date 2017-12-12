# path with l1- logistic regression

from sklearn.datasets import make_classification
from sklearn import linear_model
from sklearn import datasets
from sklearn.svm import l1_min_c
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data
y = iris.target

# a = pd.read_csv('sample20170117_labeled_0207.csv')
# x = a.values[0: 100, 0: 110]
# y = a.values[0: 100, 110]

# x, y = make_classification(n_samples=100, n_features=10, n_classes=2)

cs = l1_min_c(x, y, loss='log') * np.logspace(0, 3)
clf = linear_model.LogisticRegression(penalty='l1')
coefs_ = []
for c in cs:
    clf.set_params(C=c)
    clf.fit(x, y)
    coefs_.append(clf.coef_.ravel().copy())

coefs_ = np.array(coefs_)
plt.plot(np.log10(cs), coefs_)
ymin, ymax = plt.ylim()
plt.xlabel('log(c)')
plt.ylabel('coefficients')
plt.title('logistic regression path')
plt.axis('tight')
plt.show()

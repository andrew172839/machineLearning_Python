import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# import pandas as pd
#
# a = pd.read_csv('sample20170117_labeled_0207.csv')
# X = a.values[0: 100, 0: 110]
# y = a.values[0: 100, 110]
# y = np.array([1 if i == 1. else -1 for i in y])

# from sklearn import datasets
#
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# from sklearn.datasets import make_classification
#
# X, y = make_classification(n_samples=1000, n_features=100, n_classes=2)

n_alphas = 100
alphas = np.logspace(-10, -2, n_alphas)
clf = linear_model.Ridge(fit_intercept=False)

coefs = []
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X, y)
    coefs.append(clf.coef_)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('ridge coefficients')
plt.axis('tight')
plt.show()

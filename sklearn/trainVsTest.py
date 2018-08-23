import matplotlib.pyplot as plt
from sklearn import linear_model

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# n_samples_train, n_samples_test, n_features = 75, 25, 110
# np.random.seed(0)
# coef = np.random.randn(n_features)
# coef[50:] = 0.0
# X = np.random.randn(n_samples_train + n_samples_test, n_features)
# y = np.dot(X, coef)
# X_train, X_test = X[:n_samples_train], X[n_samples_train:]
# y_train, y_test = y[:n_samples_train], y[n_samples_train:]

# a = pd.read_csv('sample20170117_labeled_0207.csv')
# X = a.values[0: 100, 0: 110]
# y = a.values[0: 100, 110]
# y = np.array([1 if i == 1. else -1 for i in y])

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

X, y = make_classification(n_samples=1000, n_features=100, n_classes=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

alphas = np.logspace(-5, 1, 60)
enet = linear_model.ElasticNet(l1_ratio=0.7)
trainErrors = list()
testErrors = list()
for alpha in alphas:
    enet.set_params(alpha=alpha)
    enet.fit(X_train, y_train)
    trainErrors.append(enet.score(X_train, y_train))
    testErrors.append(enet.score(X_test, y_test))

i_alpha_optimal = np.argmax(testErrors)
alpha_optimal = alphas[i_alpha_optimal]
print("optimal regularization parameter, %s" % alpha_optimal)

enet.set_params(alpha=alpha_optimal)
coef_ = enet.fit(X, y).coef_

plt.subplot(2, 1, 1)
plt.semilogx(alphas, trainErrors, label='train')
plt.semilogx(alphas, testErrors, label='test')
plt.vlines(alpha_optimal, plt.ylim()[0], np.max(testErrors), label='optimum on test')
plt.legend(loc='lower left')
plt.ylim([0, 1.2])
plt.xlabel('regularization parameter')
plt.ylabel('performance')

plt.subplot(2, 1, 2)
# plt.plot(coef, label='true coef')
plt.plot(coef_, label='estimated coef')
plt.legend()
plt.show()

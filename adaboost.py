import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier

import pandas as pd
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

n_estimators = 400
learning_rate = 1.

X, y = datasets.make_hastie_10_2(n_samples=12000)

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# X, y = make_classification(n_samples=1000, n_features=100, n_classes=2)

# a = pd.read_csv('sample20170117_labeled_0207.csv')
# X = a.values[0: 100, 0: 110]
# y = a.values[0: 100, 110]
# y = np.array([1 if i == 1. else -1 for i in y])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(X_train, y_train)
dt_stump_err = 1.0 - dt_stump.score(X_test, y_test)

dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
dt.fit(X_train, y_train)
dt_err = 1.0 - dt.score(X_test, y_test)

ada_discrete = AdaBoostClassifier(base_estimator=dt_stump, learning_rate=learning_rate, n_estimators=n_estimators,
                                  algorithm="SAMME")
ada_discrete.fit(X_train, y_train)

ada_real = AdaBoostClassifier(base_estimator=dt_stump, learning_rate=learning_rate, n_estimators=n_estimators,
                              algorithm="SAMME.R")
ada_real.fit(X_train, y_train)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot([1, n_estimators], [dt_stump_err] * 2, 'k-', label='decision stump error')
ax.plot([1, n_estimators], [dt_err] * 2, 'k--', label='decision tree error')

ada_discrete_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(X_test)):
    ada_discrete_err[i] = zero_one_loss(y_pred, y_test)

ada_discrete_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(X_train)):
    ada_discrete_err_train[i] = zero_one_loss(y_pred, y_train)

ada_real_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(X_test)):
    ada_real_err[i] = zero_one_loss(y_pred, y_test)

ada_real_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(X_train)):
    ada_real_err_train[i] = zero_one_loss(y_pred, y_train)

ax.plot(np.arange(n_estimators) + 1, ada_discrete_err, label='discrete adaBoost test error', color='red')
ax.plot(np.arange(n_estimators) + 1, ada_discrete_err_train, label='discrete adaBoost train error', color='blue')
ax.plot(np.arange(n_estimators) + 1, ada_real_err, label='real adaBoost test error', color='orange')
ax.plot(np.arange(n_estimators) + 1, ada_real_err_train, label='real AdaBoost train error', color='green')

ax.set_ylim((0.0, 0.5))
ax.set_xlabel('n_estimators')
ax.set_ylabel('error rate')

leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)

plt.show()

import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import datasets

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)
X = X.astype(np.float32)
labels, y = np.unique(y, return_inverse=True)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

# a = pd.read_csv('sample20170117_labeled_0207.csv')
# X = a.values[0: 100, 0: 110]
# y = a.values[0: 100, 110]
# y = np.array([1 if i == 1. else -1 for i in y])

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# X, y = make_classification(n_samples=1000, n_features=100, n_classes=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

original_params = {'n_estimators': 1000, 'max_leaf_nodes': 4, 'max_depth': None, 'min_samples_split': 5}

plt.figure()

for label, color, setting in [('No shrinkage', 'orange', {'learning_rate': 1.0, 'subsample': 1.0}),
                              ('learning_rate=0.1', 'turquoise', {'learning_rate': 0.1, 'subsample': 1.0}),
                              ('subsample=0.5', 'blue', {'learning_rate': 1.0, 'subsample': 0.5}),
                              ('learning_rate=0.1, subsample=0.5', 'gray', {'learning_rate': 0.1, 'subsample': 0.5}),
                              ('learning_rate=0.1, max_features=2', 'magenta',
                               {'learning_rate': 0.1, 'max_features': 2})]:
    params = dict(original_params)
    params.update(setting)

    clf = ensemble.GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)

    test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        test_deviance[i] = clf.loss_(y_test, y_pred)

    plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5], '-', color=color, label=label)

plt.legend(loc='upper left')
plt.xlabel('boosting iterations')
plt.ylabel('test set deviance')

plt.show()

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# digits = load_digits()
# X, y = digits.data, digits.target

# a = pd.read_csv('sample20170117_labeled_0207.csv')
# X = a.values[0: 100, 0: 110]
# y = a.values[0: 100, 110]
# y = np.array([1 if i == 1. else -1 for i in y])

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

X, y = make_classification(n_samples=1000, n_features=100, n_classes=2)

param_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(SVC(), X, y, param_name="gamma", param_range=param_range, cv=5,
                                             scoring="accuracy")
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.xlabel("$\gamma$")
plt.ylabel('score')
plt.ylim(0.0, 1.1)
plt.semilogx(param_range, train_scores_mean, label='training')
plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std)
plt.semilogx(param_range, test_scores_mean, label='cv')
plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std)
plt.show()

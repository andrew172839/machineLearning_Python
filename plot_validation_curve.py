# plotting validation curves

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.datasets import make_classification

# digits = load_digits()
# X, y = digits.data, digits.target

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=0)

param_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(SVC(), X, y, param_name="gamma", param_range=param_range, cv=10,
                                             scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title('validation curve with svm')
plt.xlabel("$\gamma$")
plt.ylabel('score')
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label='training score', color='darkorange', lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2,
                 color='darkorange', lw=lw)
plt.semilogx(param_range, test_scores_mean, label='cross-validation score', color='navy', lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2,
                 color='navy', lw=lw)
plt.legend(loc='best')
plt.show()

import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# a = pd.read_csv('sample20170117_labeled_0207.csv')
# X = a.values[0: 100, 0: 110]
# y = a.values[0: 100, 110]
# y = np.array([1 if i == 1. else -1 for i in y])

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

X, y = make_classification(n_samples=1000, n_features=100, n_classes=2)

lasso = Lasso()
alphas = np.logspace(-4, -0.5, 30)

scores = list()
scores_std = list()

n_folds = 10

for alpha in alphas:
    lasso.alpha = alpha
    this_scores = cross_val_score(lasso, X, y, cv=n_folds, n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))

scores, scores_std = np.array(scores), np.array(scores_std)

plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('cv score')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])

lasso_cv = LassoCV(alphas=alphas)
k_fold = KFold(3)

for k, (train, test) in enumerate(k_fold.split(X, y)):
    lasso_cv.fit(X[train], y[train])
    print("[fold {0}] alpha, {1:.5f}, score, {2:.5f}".format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))

plt.show()

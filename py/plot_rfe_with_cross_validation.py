"""
===================================================
Recursive feature elimination with cross-validation
===================================================

A recursive feature elimination example with automatic tuning of the
number of features selected with cross-validation.
"""
print(__doc__)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Build a classification task using 3 informative features
#X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
#                           n_redundant=2, n_repeated=0, n_classes=8,
#                           n_clusters_per_class=1, random_state=0)

a = pd.read_csv('sample20170117_labeled_0207.csv')
X = a.values[0: 200, 0: 110]
y = a.values[0: 200, 110]
y = np.array([1 if i == 1. else -1 for i in y])

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

"""
=====================
Lasso path using LARS
=====================

Computes Lasso Path along the regularization parameter using the LARS
algorithm on the diabetes dataset. Each color represents a different
feature of the coefficient vector, and this is displayed as a function
of the regularization parameter.

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import datasets

#diabetes = datasets.load_diabetes()
#X = diabetes.data
#y = diabetes.target

import pandas as pd
a = pd.read_csv('sample20170117_labeled_0207.csv')
X = a.values[:, 0: 110]
y = a.values[:, 110]
y = np.array([1 if i == 1. else -1 for i in y])

print("Computing regularization path using the LARS ...")
alphas, _, coefs = linear_model.lars_path(X, y, method = 'lasso', verbose = True)

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle = 'dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.show()
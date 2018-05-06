import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from sklearn import datasets
from sklearn.datasets import make_classification
import pandas as pd

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# X, y = make_classification(n_samples=1000, n_features=100, n_classes=2)

# a = pd.read_csv('sample20170117_labeled_0207.csv')
# X = a.values[0: 100, 0: 110]
# y = a.values[0: 100, 110]
# y = np.array([1 if i == 1. else -1 for i in y])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

alphas, _, coefs = linear_model.lars_path(X_train, y_train, method='lasso', verbose=True)

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('coefficients')
plt.title('lasso path')
plt.axis('tight')
plt.show()

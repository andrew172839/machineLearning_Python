import matplotlib.pyplot as plt
from sklearn import linear_model

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data[:, np.newaxis, 2]
X_train = X[: -20]
X_test = X[-20:]
y_train = diabetes.target[: -20]
y_test = diabetes.target[-20:]

# a = pd.read_csv('sample20170117_labeled_0207.csv')
# X = a.values[0: 100, 0: 110]
# y = a.values[0: 100, 110]
# y = np.array([1 if i == 1. else -1 for i in y])

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# X, y = make_classification(n_samples=1000, n_features=100, n_classes=2)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regression = linear_model.LinearRegression()
regression.fit(X_train, y_train)
print('coefficients, \n', regression.coef_)
print("mean squared error, %.2f" % np.mean((regression.predict(X_test) - y_test) ** 2))
print('variance, %.2f' % regression.score(X_test, y_test))

plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, regression.predict(X_test), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

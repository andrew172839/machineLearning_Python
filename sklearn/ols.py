import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

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
print('X_train, \n', X_train)
print('y_train, \n', y_train)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print('coefficients, \n', regr.coef_)
print("mean squared error, %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2))
print('variance, %.2f' % regr.score(X_test, y_test))

plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, regr.predict(X_test), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

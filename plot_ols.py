import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
# y = np.ones(10)
# diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(X, y, test_size=0.2,
#                                                                                         random_state=0)

# import pandas as pd
#
# a = pd.read_csv('sample20170117_labeled_0207.csv')
# X = a.values[0: 100, 0: 110]
# y = a.values[0: 100, 110]
# y = np.array([1 if i == 1. else -1 for i in y])
# diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(X, y, test_size=0.2,
#                                                                                         random_state=0)

regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
diabetes_x = diabetes.data[:, np.newaxis, 2]
diabetes_x_train = diabetes_x[: -20]
diabetes_x_test = diabetes_x[-20:]
diabetes_y_train = diabetes.target[: -20]
diabetes_y_test = diabetes.target[-20:]

regr = linear_model.LinearRegression()
regr.fit(diabetes_x_train, diabetes_y_train)
print('coefficients, \n', regr.coef_)
print("mean squared error, %.2f" % np.mean((regr.predict(diabetes_x_test) - diabetes_y_test) ** 2))
print('variance, %.2f' % regr.score(diabetes_x_test, diabetes_y_test))

plt.scatter(diabetes_x_test, diabetes_y_test, color='black')
plt.plot(diabetes_x_test, regr.predict(diabetes_x_test), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

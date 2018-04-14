import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.model_selection import train_test_split

a = pd.read_csv('C:/Users/yushao/Documents/GitHub/machineLearning_Python/sample20170117_labeled_0207.csv')

X = a.values[0: 100, 0: 110]
y = a.values[0: 100, 110]
y = np.array([1 if i == 1. else -1 for i in y])

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print('train')
print(X_train)
print('test')
print(X_test)

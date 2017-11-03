import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

# iris = sns.load_dataset("iris")
# iris.head()
# X = iris.values[:, :4]
# y = iris.values[:, 4]

import pandas as pd

a = pd.read_csv('sample20170117_labeled_0207.csv')
X = a.values[0: 50, 0: 110]
y = a.values[0: 50, 110]
y = np.array([1 if i == 1. else -1 for i in y])

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
lr = LogisticRegressionCV()
lr.fit(train_X, train_y)
print("Accuracy = {:.2f}".format(lr.score(test_X, test_y)))

def one_hot_encode_object_array(arr):
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

train_y_ohe = one_hot_encode_object_array(train_y)
test_y_ohe = one_hot_encode_object_array(test_y)

model = Sequential()

# model.add(Dense(16, input_shape=(4,)))
model.add(Dense(16, input_shape=(110,)))
model.add(Activation('sigmoid'))

# model.add(Dense(3))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_X, train_y_ohe, nb_epoch=100, batch_size=1, verbose=0);
loss, accuracy = model.evaluate(test_X, test_y_ohe, verbose=0)
print("Accuracy = {:.2f}".format(accuracy))
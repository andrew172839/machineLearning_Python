from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import metrics
from sklearn import preprocessing
import tensorflow as tf
import numpy as np

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def main(unused_argv):
    boston = tf.contrib.learn.datasets.load_dataset('boston')
    X, y = boston.data, boston.target

    # a = pd.read_csv('sample20170117_labeled_0207.csv')
    # X = a.values[0: 100, 0: 110]
    # y = a.values[0: 100, 110]
    # y = np.array([1 if i == 1. else -1 for i in y])

    # X, y = make_classification(n_samples=1000, n_features=100, n_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)

    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns, hidden_units=[10, 10])
    regressor.fit(X_train, y_train, steps=5000, batch_size=1)
    y_predicted = list(regressor.predict(scaler.transform(X_test), as_iterable=True))
    score = metrics.mean_squared_error(y_predicted, y_test)
    print('mse, {0: f}'.format(score))
    print(y_predicted);


if __name__ == '__main__':
    tf.app.run()

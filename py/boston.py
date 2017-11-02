from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import cross_validation
from sklearn import metrics
from sklearn import preprocessing
from sklearn.datasets import make_classification
import tensorflow as tf
import pandas as pd
import numpy as np

def main(unused_argv):
    # boston = tf.contrib.learn.datasets.load_dataset('boston')
    # x, y = boston.data, boston.target

    # a = pd.read_csv('sample20170117_labeled_0207.csv')
    # x = a.values[:, 0: 110]
    # y = a.values[:, 110]
    # y = np.array([1 if i == 1. else -1 for i in y])

    # x, y = make_classification(n_samples=1000, n_features=100, n_classes=2)

    x = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
    y = np.ones(10)

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)

    # build 2 layer fully connected dnns with 10, 10 units respectively
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train)
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns, hidden_units=[10, 10])
    regressor.fit(x_train, y_train, steps=5000, batch_size=1)
    y_predicted = list(regressor.predict(scaler.transform(x_test), as_iterable=True))
    score = metrics.mean_squared_error(y_predicted, y_test)
    print('mse: {0: f}'.format(score))

if __name__ == '__main__':
    tf.app.run()

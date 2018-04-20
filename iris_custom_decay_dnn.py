from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import tensorflow as tf

from sklearn import datasets
from sklearn.datasets import make_classification
import pandas as pd

import numpy as np


def optimizer_exp_decay():
    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step, decay_steps=100,
                                               decay_rate=0.001)
    return tf.train.AdagradOptimizer(learning_rate=learning_rate)


def main(unused_argv):
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # import pandas as pd
    # a = pd.read_csv('sample20170117_labeled_0207.csv')
    # X = a.values[0: 100, 0: 110]
    # y = a.values[0: 100, 110]
    # y = np.array([1 if i == 1. else -1 for i in y])
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X, y = make_classification(n_samples=1000, n_features=100, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=2,
                                                optimizer=optimizer_exp_decay)

    classifier.fit(X_train, y_train, steps=800)
    predictions = list(classifier.predict(X_test, as_iterable=True))
    score = metrics.accuracy_score(y_test, predictions)
    print('accuracy, {0:f}'.format(score))


if __name__ == '__main__':
    tf.app.run()

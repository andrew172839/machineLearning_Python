from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import metrics
import shutil
import tensorflow as tf

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

learn = tf.contrib.learn


def clean_folder(folder):
    try:
        shutil.rmtree(folder)
    except OSError:
        pass


def main(unused_argv):
    # a = pd.read_csv('sample20170117_labeled_0207.csv')
    # X = a.values[0: 100, 0: 110]
    # y = a.values[0: 100, 110]
    # y = np.array([1 if i == 1. else -1 for i in y])

    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target

    X, y = make_classification(n_samples=1000, n_features=100, n_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    val_monitor = learn.monitors.ValidationMonitor(X_val, y_val, early_stopping_rounds=200)

    classifier1 = learn.DNNClassifier(
        feature_columns=learn.infer_real_valued_columns_from_input(X_train),
        hidden_units=[10, 20, 10],
        n_classes=2)
    classifier1.fit(x=X_train, y=y_train, steps=2000)
    predictions1 = list(classifier1.predict(X_test, as_iterable=True))
    score1 = metrics.accuracy_score(y_test, predictions1)

    classifier2 = learn.DNNClassifier(
        feature_columns=learn.infer_real_valued_columns_from_input(X_train),
        hidden_units=[10, 20, 10],
        n_classes=2,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))
    classifier2.fit(x=X_train, y=y_train, steps=2000, monitors=[val_monitor])
    predictions2 = list(classifier2.predict(X_test, as_iterable=True))
    score2 = metrics.accuracy_score(y_test, predictions2)

    print('score1, ', score1)
    print('score2, ', score2)
    print('score2 > score1, ', score2 > score1)


if __name__ == '__main__':
    tf.app.run()

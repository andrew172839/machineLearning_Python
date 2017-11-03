from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification

learn = tf.contrib.learn


def clean_folder(folder):
    try:
        shutil.rmtree(folder)
    except OSError:
        pass


def main(unused_argv):
    # iris = datasets.load_iris()

    # a = pd.read_csv('sample20170117_labeled_0207.csv')
    # x = a.values[:, 0: 110]
    # y = a.values[:, 110]
    # y = np.array([1 if i == 1. else -1 for i in y])

    x, y = make_classification(n_samples=1000, n_features=100, n_classes=2)

    # x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    val_monitor = learn.monitors.ValidationMonitor(x_val, y_val, early_stopping_rounds=200)

    # model_dir = '/tmp/iris_model'
    model_dir = 'C:/Users/yushao/PycharmProjects/hw/model'
    clean_folder(model_dir)

    classifier1 = learn.DNNClassifier(
        feature_columns=learn.infer_real_valued_columns_from_input(x_train),
        hidden_units=[10, 20, 10],
        n_classes=2,
        model_dir=model_dir)
    classifier1.fit(x=x_train, y=y_train, steps=2000)
    predictions1 = list(classifier1.predict(x_test, as_iterable=True))
    score1 = metrics.accuracy_score(y_test, predictions1)

    # model_dir = '/tmp/iris_model_val'
    model_dir = 'C:/Users/yushao/PycharmProjects/hw/model_val'
    clean_folder(model_dir)

    # classifier with early stopping on validation data, save frequently for monitor to pick up new checkpoints.
    classifier2 = learn.DNNClassifier(
        feature_columns=learn.infer_real_valued_columns_from_input(x_train),
        hidden_units=[10, 20, 10],
        n_classes=3,
        model_dir=model_dir,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))
    classifier2.fit(x=x_train, y=y_train, steps=2000, monitors=[val_monitor])
    predictions2 = list(classifier2.predict(x_test, as_iterable=True))
    score2 = metrics.accuracy_score(y_test, predictions2)

    print('score1: ', score1)
    print('score2: ', score2)
    print('score2 > score1: ', score2 > score1)


if __name__ == '__main__':
    tf.app.run()

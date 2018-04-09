from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import datasets
#from sklearn import metrics
#from sklearn.cross_validation import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np


def optimizer_exp_decay():
    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step, decay_steps=100,
                                               decay_rate=0.001)
    return tf.train.AdagradOptimizer(learning_rate=learning_rate)


def main(unused_argv):
    iris = datasets.load_iris()

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

    # a = pd.read_csv('sample20170117_labeled_0207.csv')
    # training = a.values[:, 0: 110]
    # label = a.values[:, 110]

    # x_train, x_test, y_train, y_test = train_test_split(training, label, test_size=0.2, random_state=42)

    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train)
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=3,
                                                optimizer=optimizer_exp_decay)

    classifier.fit(x_train, y_train, steps=800)
    predictions = list(classifier.predict(x_test, as_iterable=True))
    score = metrics.accuracy_score(y_test, predictions)
    print('accuracy, {0:f}'.format(score))


if __name__ == '__main__':
    tf.app.run()

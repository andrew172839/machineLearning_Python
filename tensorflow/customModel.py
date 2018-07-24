from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import metrics
import tensorflow as tf

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

layers = tf.contrib.layers
learn = tf.contrib.learn


def my_model(features, target):
    target = tf.one_hot(target, 2, 1, 0)

    normalizer_fn = layers.dropout
    normalizer_params = {'keep_prob': 0.9}
    features = layers.stack(features, layers.fully_connected, [10, 20, 10], normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params)

    logits = layers.fully_connected(features, 2, activation_fn=None)
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(), optimizer='Adam',
                                               learning_rate=0.1)
    return ({'class': tf.argmax(logits, 1), 'prob': tf.nn.softmax(logits)}, loss, train_op)


def main(unused_argv):
    # a = pd.read_csv('sample20170117_labeled_0207.csv')
    # X = a.values[0: 100, 0: 110]
    # y = a.values[0: 100, 110]
    # y = np.array([1 if i == 1. else -1 for i in y])

    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target

    X, y = make_classification(n_samples=100, n_features=10, n_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    classifier = learn.Estimator(model_fn=my_model)
    classifier.fit(X_train, y_train, steps=200)

    y_predicted = [p['class'] for p in classifier.predict(X_test, as_iterable=True)]
    score = metrics.accuracy_score(y_test, y_predicted)
    print('accuracy, {0: f}'.format(score))


if __name__ == '__main__':
    tf.app.run()

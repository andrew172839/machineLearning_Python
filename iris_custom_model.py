from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import cross_validation
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import numpy as np

layers = tf.contrib.layers
learn = tf.contrib.learn


def my_model(features, target):
    target = tf.one_hot(target, 3, 1, 0)

    normalizer_fn = layers.dropout
    normalizer_params = {'keep_prob': 0.9}
    features = layers.stack(
        features,
        layers.fully_connected, [10, 20, 10],
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params)

    logits = layers.fully_connected(features, 3, activation_fn=None)
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adagrad',
        learning_rate=0.1)

    return ({'class': tf.argmax(logits, 1), 'prob': tf.nn.softmax(logits)}, loss, train_op)


def main(unused_argv):
    # a = pd.read_csv('sample20170117_labeled_0207.csv')
    # x = a.values[:, 0: 110]
    # y = a.values[:, 110]
    # y = np.array([1 if i == 1. else -1 for i in y])

    # iris = datasets.load_iris()
    # x = iris.data
    # y = iris.target

    x, y = make_classification(n_samples=100, n_features=10, n_classes=2)

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        x, y)

    classifier = learn.Estimator(model_fn=my_model)
    classifier.fit(x_train, y_train, steps=1000)

    y_predicted = [p['class'] for p in classifier.predict(x_test, as_iterable=True)]
    score = metrics.accuracy_score(y_test, y_predicted)
    print('accuracy, {0: f}'.format(score))


if __name__ == '__main__':
    tf.app.run()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import cross_validation
from sklearn import datasets
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import numpy as np

layers = tf.contrib.layers
learn = tf.contrib.learn


def my_model(features, target):
    # DNN with three hidden layers, and dropout of 0.1 probability.
    # Convert the target to a one-hot tensor of shape (length of features, 3) and
    # with a on-value of 1 for each one-hot vector of length 3.
    target = tf.one_hot(target, 3, 1, 0)

    # Create three fully connected layers respectively of size 10, 20, and 10 with
    # each layer having a dropout probability of 0.1.
    normalizer_fn = layers.dropout
    normalizer_params = {'keep_prob': 0.9}
    features = layers.stack(
        features,
        layers.fully_connected, [10, 20, 10],
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params)

    # Compute logits (1 per class) and compute loss.
    logits = layers.fully_connected(features, 3, activation_fn=None)
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

    # Create a tensor for training op.
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adagrad',
        learning_rate=0.1)

    return ({'class': tf.argmax(logits, 1), 'prob': tf.nn.softmax(logits)}, loss, train_op)


def main(unused_argv):
    # iris = datasets.load_iris()

    a = pd.read_csv('sample20170117_labeled_0207.csv')
    training = a.values[:, 0: 110]
    label = a.values[:, 110]
    label = np.array([1 if i == 1. else -1 for i in label])

    # x_train, x_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.2,
    #                                                                      random_state=42)

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        training, label, test_size=0.2, random_state=42)

    classifier = learn.Estimator(model_fn=my_model)
    classifier.fit(x_train, y_train, steps=1000)

    y_predicted = [p['class'] for p in classifier.predict(x_test, as_iterable=True)]
    score = metrics.accuracy_score(y_test, y_predicted)
    print('Accuracy: {0:f}'.format(score))


if __name__ == '__main__':
    tf.app.run()

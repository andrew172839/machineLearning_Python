from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import cross_validation
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import numpy as np

def main(unused_argv):
  # Load dataset.
  #iris = tf.contrib.learn.datasets.load_dataset('iris')
  #x_train, x_test, y_train, y_test = cross_validation.train_test_split(
  #    iris.data, iris.target, test_size=0.2, random_state=42)

  a = pd.read_csv('sample20170117_labeled_0207.csv')
#a = pd.read_csv('sample20170117_labeled_01.csv')
  training = a.values[:, 0: 110]
  label = a.values[:, 110]
  label = np.array([1 if i == 1. else -1 for i in label])
  x_train, x_test, y_train, y_test = cross_validation.train_test_split(
      training, label, test_size=0.2, random_state=42)

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
      x_train)
  classifier = tf.contrib.learn.DNNClassifier(
      feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3)

  # Fit and predict.
  classifier.fit(x_train, y_train, steps=200)
  predictions = list(classifier.predict(x_test, as_iterable=True))
  score = metrics.accuracy_score(y_test, predictions)
  print('Accuracy: {0:f}'.format(score))


if __name__ == '__main__':
  tf.app.run()

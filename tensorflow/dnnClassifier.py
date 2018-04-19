import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

x, y = make_classification(n_samples=1000, n_features=100, n_classes=2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

feature_columns = [tf.contrib.layers.real_valued_column('', dimension=100)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=2)
classifier.fit(x_train, y_train, steps=200)

accuracy_score = classifier.evaluate(x_test, y_test)['accuracy']
print('accuracy, {0: f}'.format(accuracy_score))

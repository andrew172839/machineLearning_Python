from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd

a = pd.read_csv('sample20170117_labeled_0207.csv')
X = a.values[0: 50, 0: 110]
y = a.values[0: 50, 110]
y = np.array([1 if i == 1. else -1 for i in y])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

feature_columns = [tf.contrib.layers.real_valued_column('', dimension = 110)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns, hidden_units = [10, 20, 10], n_classes = 2)
classifier.fit(X_train, y_train, steps = 200)

accuracy_score = classifier.evaluate(X_test, y_test)['accuracy']
print('Accuracy: {0:f}'.format(accuracy_score))

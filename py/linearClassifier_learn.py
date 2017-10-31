import tensorflow.contrib.learn.python.learn as learn
import pandas as pd
import numpy as np
from sklearn import metrics

a = pd.read_csv('sample20170117_labeled_0207.csv')
X = a.values[0: 100, 0: 110]
y = a.values[0: 100, 110]
y = np.array([1 if i == 1. else -1 for i in y])

feature_columns = learn.infer_real_valued_columns_from_input(X)
classifier = learn.LinearClassifier(n_classes=2, feature_columns=feature_columns)
classifier.fit(X, y, steps=200, batch_size=32)
predictions = list(classifier.predict(X, as_iterable=True))
score = metrics.accuracy_score(y, predictions)
print('Accuracy: %f' % score)
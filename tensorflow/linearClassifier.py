from sklearn import metrics
from tensorflow.contrib.learn.python import learn

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# a = pd.read_csv('sample20170117_labeled_0207.csv')
# X = a.values[0: 100, 0: 110]
# y = a.values[0: 100, 110]
# y = np.array([1 if i == 1. else -1 for i in y])

iris = datasets.load_iris()
X = iris.data
y = iris.target

# X, y = make_classification(n_samples=1000, n_features=100, n_classes=2)

feature_columns = learn.infer_real_valued_columns_from_input(X)
classifier = learn.LinearClassifier(n_classes=3, feature_columns=feature_columns)
classifier.fit(X, y, steps=200, batch_size=32)
predictions = list(classifier.predict(X, as_iterable=True))
score = metrics.accuracy_score(y, predictions)
print('accuracy, %f' % score)

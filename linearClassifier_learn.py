from sklearn import metrics
from tensorflow.contrib.learn.python import learn
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

# a = pd.read_csv('sample20170117_labeled_0207.csv')
# x = a.values[0: 100, 0: 110]
# y = a.values[0: 100, 110]
# y = np.array([1 if i == 1. else -1 for i in y])

# x = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
# y = np.ones(10)

# x, y = make_classification(n_samples=1000, n_features=100, n_classes=2)

feature_columns = learn.infer_real_valued_columns_from_input(x)
classifier = learn.LinearClassifier(n_classes=2, feature_columns=feature_columns)
classifier.fit(x, y, steps=200, batch_size=32)
predictions = list(classifier.predict(x, as_iterable=True))
score = metrics.accuracy_score(y, predictions)
print('accuracy, %f' % score)

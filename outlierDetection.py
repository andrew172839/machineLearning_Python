import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.datasets import load_boston
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM

import pandas as pd
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# a = pd.read_csv('sample20170117_labeled_0207.csv')
# X = a.values[0: 100, 0: 110]
# y = a.values[0: 100, 110]
# y = np.array([1 if i == 1. else -1 for i in y])

# X1, X2, y1, y2 = train_test_split(X, y, test_size=0.2)

X1 = load_boston()['data'][:, [8, 10]]
X2 = load_boston()['data'][:, [5, 12]]

classifiers = \
    {"empirical covariance": EllipticEnvelope(support_fraction=1., contamination=0.261),
     "robust covariance (minimum covariance determinant)": EllipticEnvelope(contamination=0.261),
     "ocsvm": OneClassSVM(nu=0.261, gamma=0.05)}
colors = ['m', 'g', 'b']
legend1 = {}
legend2 = {}

xx1, yy1 = np.meshgrid(np.linspace(-8, 28, 500), np.linspace(3, 40, 500))
xx2, yy2 = np.meshgrid(np.linspace(3, 10, 500), np.linspace(-5, 45, 500))
for i, (clf_name, clf) in enumerate(classifiers.items()):
    plt.figure(1)
    clf.fit(X1)
    Z1 = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
    Z1 = Z1.reshape(xx1.shape)
    legend1[clf_name] = plt.contour(xx1, yy1, Z1, levels=[0], linewidths=2, colors=colors[i])
    plt.figure(2)
    clf.fit(X2)
    Z2 = clf.decision_function(np.c_[xx2.ravel(), yy2.ravel()])
    Z2 = Z2.reshape(xx2.shape)
    legend2[clf_name] = plt.contour(xx2, yy2, Z2, levels=[0], linewidths=2, colors=colors[i])

legend1_values_list = list(legend1.values())
legend1_keys_list = list(legend1.keys())

plt.figure(1)
plt.title("outlier detection on a real data set")
plt.scatter(X1[:, 0], X1[:, 1], color='black')
bbox_args = dict(boxstyle="round", fc="0.8")
arrow_args = dict(arrowstyle="->")
plt.annotate("several confounded points", xy=(24, 19), xycoords="data", textcoords="data", xytext=(13, 10),
             bbox=bbox_args, arrowprops=arrow_args)
plt.xlim((xx1.min(), xx1.max()))
plt.ylim((yy1.min(), yy1.max()))
plt.legend((legend1_values_list[0].collections[0], legend1_values_list[1].collections[0],
            legend1_values_list[2].collections[0]), (legend1_keys_list[0], legend1_keys_list[1], legend1_keys_list[2]),
           loc="upper center", prop=matplotlib.font_manager.FontProperties(size=12))
plt.ylabel("accessibility to radial highways")
plt.xlabel("pupil-teacher ratio by town")

legend2_values_list = list(legend2.values())
legend2_keys_list = list(legend2.keys())

plt.figure(2)
plt.title("outlier detection on a real data set")
plt.scatter(X2[:, 0], X2[:, 1], color='black')
plt.xlim((xx2.min(), xx2.max()))
plt.ylim((yy2.min(), yy2.max()))
plt.legend((legend2_values_list[0].collections[0], legend2_values_list[1].collections[0],
            legend2_values_list[2].collections[0]), (legend2_keys_list[0], legend2_keys_list[1], legend2_keys_list[2]),
           loc="upper center", prop=matplotlib.font_manager.FontProperties(size=12))
plt.ylabel("% lower status of the population")
plt.xlabel("average number of rooms per dwelling")
plt.show()

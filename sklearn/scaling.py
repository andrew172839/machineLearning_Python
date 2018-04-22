from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn import datasets
from sklearn.datasets import make_classification
import pandas as pd

# np.random.seed(42)
# n_datapoints = 100
# Cov = [[0.9, 0.0], [0.0, 20.0]]
# mu1 = [100.0, -3.0]
# mu2 = [101.0, -3.0]
#
# X1 = np.random.multivariate_normal(mean=mu1, cov=Cov, size=n_datapoints)
# X2 = np.random.multivariate_normal(mean=mu2, cov=Cov, size=n_datapoints)
# y_train = np.hstack([[-1] * n_datapoints, [1] * n_datapoints])
# X_train = np.vstack([X1, X2])
#
# X1 = np.random.multivariate_normal(mean=mu1, cov=Cov, size=n_datapoints)
# X2 = np.random.multivariate_normal(mean=mu2, cov=Cov, size=n_datapoints)
# y_test = np.hstack([[-1] * n_datapoints, [1] * n_datapoints])
# X_test = np.vstack([X1, X2])
#
# X_train[0, 0] = -1000

X, y = make_classification(n_samples=1000, n_features=100, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# a = pd.read_csv('sample20170117_labeled_0207.csv')
# X = a.values[0: 100, 0: 110]
# y = a.values[0: 100, 110]
# y = np.array([1 if i == 1. else -1 for i in y])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

standard_scaler = StandardScaler()
Xtr_s = standard_scaler.fit_transform(X_train)
Xte_s = standard_scaler.transform(X_test)

robust_scaler = RobustScaler()
Xtr_r = robust_scaler.fit_transform(X_train)
Xte_r = robust_scaler.transform(X_test)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].scatter(X_train[:, 0], X_train[:, 1], color=np.where(y_train > 0, 'r', 'b'))
ax[1].scatter(Xtr_s[:, 0], Xtr_s[:, 1], color=np.where(y_train > 0, 'r', 'b'))
ax[2].scatter(Xtr_r[:, 0], Xtr_r[:, 1], color=np.where(y_train > 0, 'r', 'b'))
ax[0].set_title("unscaled data")
ax[1].set_title("after standard scaling (zoomed in)")
ax[2].set_title("after robust scaling (zoomed in)")
for a in ax[1:]:
    a.set_xlim(-3, 3)
    a.set_ylim(-3, 3)
plt.tight_layout()
plt.show()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(Xtr_s, Y_train)
acc_s = knn.score(Xte_s, Y_test)
print("test accuracy using standard scaler, %.3f" % acc_s)
knn.fit(Xtr_r, Y_train)
acc_r = knn.score(Xte_r, Y_test)
print("test accuracy using robust scaler, %.3f" % acc_r)

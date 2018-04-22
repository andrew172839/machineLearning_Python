from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets

from sklearn import datasets
from sklearn.datasets import make_classification
import pandas as pd

# diabetes = datasets.load_diabetes()
# X = diabetes.data
# y = diabetes.target

X, y = make_classification(n_samples=1000, n_features=100, n_classes=2)

# a = pd.read_csv('sample20170117_labeled_0207.csv')
# X = a.values[:, 0: 110]
# y = a.values[:, 110]
# y = np.array([1 if i == 1. else -1 for i in y])

X /= X.std(axis=0)

eps = 5e-3

print("computing regularization path using the lasso ...")
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=False)

print("computing regularization path using the positive lasso ...")
alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(X, y, eps, positive=True, fit_intercept=False)
print("computing regularization path using the elastic net ...")
alphas_enet, coefs_enet, _ = enet_path(X, y, eps=eps, l1_ratio=0.8, fit_intercept=False)

print("computing regularization path using the positive elastic net ...")
alphas_positive_enet, coefs_positive_enet, _ = enet_path(X, y, eps=eps, l1_ratio=0.8, positive=True,
                                                         fit_intercept=False)

plt.figure(1)
ax = plt.gca()

colors = cycle(['b', 'r', 'g', 'c', 'k'])
neg_log_alphas_lasso = -np.log10(alphas_lasso)
neg_log_alphas_enet = -np.log10(alphas_enet)
for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2 = plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)

plt.xlabel('-log(alpha)')
plt.ylabel('coefficients')
plt.title('lasso and elasticNet paths')
plt.legend((l1[-1], l2[-1]), ('lasso', 'elasticNet'), loc='lower left')
plt.axis('tight')

plt.figure(2)
ax = plt.gca()
neg_log_alphas_positive_lasso = -np.log10(alphas_positive_lasso)
for coef_l, coef_pl, c in zip(coefs_lasso, coefs_positive_lasso, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2 = plt.plot(neg_log_alphas_positive_lasso, coef_pl, linestyle='--', c=c)

plt.xlabel('-log(alpha)')
plt.ylabel('coefficients')
plt.title('lasso and positive lasso')
plt.legend((l1[-1], l2[-1]), ('lasso', 'positive Lasso'), loc='lower left')
plt.axis('tight')

plt.figure(3)
ax = plt.gca()
neg_log_alphas_positive_enet = -np.log10(alphas_positive_enet)
for (coef_e, coef_pe, c) in zip(coefs_enet, coefs_positive_enet, colors):
    l1 = plt.plot(neg_log_alphas_enet, coef_e, c=c)
    l2 = plt.plot(neg_log_alphas_positive_enet, coef_pe, linestyle='--', c=c)

plt.xlabel('-log(alpha)')
plt.ylabel('coefficients')
plt.title('elasticNet and positive elasticNet')
plt.legend((l1[-1], l2[-1]), ('elasticNet', 'positive elasticNet'), loc='lower left')
plt.axis('tight')
plt.show()

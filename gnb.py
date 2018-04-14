import numpy as np

# create a grid on which to evaluate the distributions
grid = np.linspace(-3, 6, 100)
xgrid, ygrid = np.meshgrid(grid, grid)
Xgrid = np.vstack([xgrid.ravel(), ygrid.ravel()]).T

# now evaluate and plot the probability grid
prob_grid = np.exp(gnb._joint_log_likelihood(Xgrid))
for i, c in enumerate(['blue', 'red']):
    pl.contour(xgrid, ygrid, prob_grid[:, i].reshape((100, 100)), 3, colors=c)

# plot the points as above# create a grid on which to evaluate the distributions
grid = np.linspace(-3, 6, 100)
xgrid, ygrid = np.meshgrid(grid, grid)
Xgrid = np.vstack([xgrid.ravel(), ygrid.ravel()]).T

# now evaluate and plot the probability grid
prob_grid = np.exp(gnb._joint_log_likelihood(Xgrid))
for i, c in enumerate(['blue', 'red']):
    pl.contour(xgrid, ygrid, prob_grid[:, i].reshape((100, 100)), 3, colors=c)

pl.scatter(X[:, 0], X[:, 1], c=y, linewidth=0)
pl.scatter(X[:, 0], X[:, 1], c=y, linewidth=0)

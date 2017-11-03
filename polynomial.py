x_test = np.linspace(-1, 1, 100)

pl.figure()
pl.scatter(x, y, s=4)

X = np.array([x ** i for i in range(5)]).T
X_test = np.array([x_test ** i for i in range(5)]).T
order4 = LinearRegression()
order4.fit(X, y)
pl.plot(x_test, order4.predict(X_test), label='4th order')

X = np.array([x ** i for i in range(10)]).T
X_test = np.array([x_test ** i for i in range(10)]).T
order9 = LinearRegression()
order9.fit(X, y)
pl.plot(x_test, order9.predict(X_test), label='9th order')

pl.legend(loc='best')
pl.axis('tight')
pl.title('Fitting a 4th and a 9th order polynomial')

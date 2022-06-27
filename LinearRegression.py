"""
	LinearRegression.py
	Created by Adam Kohl
	2.1.2020

	This file contains an example of using the sklearn module
	to produce a linear regression model.
"""
from PlotGenerator import *
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor


# Example data generation
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)


def normal_equation():
	# Model initialization & fit
	model = LinearRegression()
	model.fit(X, y)

	# Model prediction on new data
	X_new = np.array([[0], [2]])
	y_predict = model.predict(X_new)

	# Plot the results
	p = ScatterPlot()
	p.add_series([X, y], 'dataset series', marker_shape='x', plt_color='b')
	p.add_series_line([X_new, y_predict], 'X new & Predicted Y', plt_color='r')
	p.set_labels('X', 'Y')
	p.set_title('Linear Regression Example - Normal Equation\n\nIntercept: {} | Coef: {}'.format(model.intercept_, model.coef_))
	p.show_legend()
	p.save_figure()
	p.show_plot()


def stochastic_gradient_descent():
	"""
		max_iter: number of epochs to run
		penalty: none indicates no use of any regularization
		eta0: learning rate
	"""
	model = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
	model.fit(X, y.ravel())

	# Model prediction on new data
	X_new = np.array([[0], [2]])
	y_predict = model.predict(X_new)

	# Plot the results
	p = ScatterPlot()
	p.add_series([X, y], 'dataset series', marker_shape='x', plt_color='b')
	p.add_series_line([X_new, y_predict], 'X new & Predicted Y', plt_color='r')
	p.set_labels('X', 'Y')
	p.set_title('Linear Regression Example - Stochastic Gradient Descent\n\nIntercept: {} | Coef: {}'.format(model.intercept_, model.coef_))
	p.show_legend()
	p.save_figure()
	p.show_plot()


# def polynomial_regression():



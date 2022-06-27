"""
	ExamplePlots.py
	Created by Adam Kohl
	2.1.2020

	This script outlines example use cases.
	All plots are generated using classes contained in PlotGenerator.py
"""
from PlotGenerator import *

""" -------------------------------------------------------------------------------
	-------------------------------- BASIC 2D PLOTS -------------------------------
	-------------------------------------------------------------------------------
"""


def line_plot(data_1, data_2):
	p = LinePlot()
	p.add_series(data_1)
	p.set_title('Line Plot | 1 Series | No Labels')
	p.show_plot()

	p = LinePlot()
	p.add_series(data_1)
	p.add_series(data_2, 'series 2')
	p.set_labels('X', 'Y')
	p.set_title('Line Plot | 2 Series | 1 Labeled Series | Labeled Axes')
	p.show_legend()
	p.show_plot()

	p = LinePlot()
	p.add_series(data_1)
	p.add_series(data_2, 'series 2')
	p.set_aiaa_specifications()
	p.set_labels('X', 'Y')
	p.set_title('Line Plot | AIAA Format')
	p.show_legend()
	p.show_plot()


def scatter_plot(data_1, data_2):
	p = ScatterPlot()
	p.add_series(data_1)
	p.set_title('Scatter Plot | 1 Series | Marker: o')
	p.show_plot()

	p = ScatterPlot()
	p.add_series(data_1, 'series 1')
	p.add_series(data_2, 'series 2', marker_shape='x')
	p.set_labels('X', 'Y')
	p.set_title('Scatter Plot | 2 Series | Markers: o & *')
	p.show_legend()
	p.show_plot()

	p = ScatterPlot()
	p.add_series(data_1, 'series 1')
	p.add_series(data_2, 'series 2', marker_shape='x', marker_size=10)
	p.set_labels('X', 'Y')
	p.set_title('Scatter Plot | 2 Series | Markers: o (size 1) & * (size 3)')
	p.show_legend()
	p.show_plot()


def histogram_plot(data_1, data_2):
	p = HistogramPlot()
	p.set_data(data_1[1], 'series 1')
	p.set_labels('Y Value', 'Frequency')
	p.set_title('Histogram Plot')
	p.show_legend()
	p.show_plot()

	p = HistogramPlot(bins=20)
	p.set_data(data_2[1], 'series 2')
	p.set_labels('Y Value', 'Frequency')
	p.set_title('Histogram Plot')
	p.show_legend()
	p.show_plot()


""" -------------------------------------------------------------------------------
	---------------------- DIMENSIONALITY REDUCTION PLOTS -------------------------
	-------------------------------------------------------------------------------
"""


def pca_plot(x, y):
	# 2D - No labels
	p = PCAPlot(2)
	p.set_data(x)
	p.set_title('Iris Dataset | 2D PCA | No Target Data')
	p.show_plot()

	# 2D - Target labels
	p = PCAPlot(2)
	p.set_data(x, y)
	p.set_title('Iris Dataset | 2D PCA | Target Data')
	p.show_plot()

	# 3D - No labels
	p = PCAPlot(3)
	p.set_data(x)
	p.set_title('Iris Dataset | 3D PCA | No Target Data')
	p.show_plot()

	# 3D - Target labels
	p = PCAPlot(3)
	p.set_data(x, y)
	p.set_title('Iris Dataset | 3D PCA | Target Data')
	p.show_plot()


def kernel_pca_plot(x, y):
	# 2D - RBF & Sigmoid
	p = PCAKernelPlot(2, 'rbf', 0.04)
	p.set_data(x, y)
	p.set_title('Iris Dataset | 2D PCA Kernel: RBF | Gamma: .04 | Target Data')
	p.show_plot()

	p = PCAKernelPlot(2, 'sigmoid', .001)
	p.set_data(x, y)
	p.set_title('Iris Dataset | 2D PCA Kernel: Sigmoid | Gamma: .001| Target Data')
	p.show_plot()

	# 3D - RBF & Sigmoid
	p = PCAKernelPlot(3, 'rbf', .04)
	p.set_data(x, y)
	p.set_title('Iris Dataset | 3D PCA Kernel: RBF | Gamma: .04 | Target Data')
	p.show_plot()

	p = PCAKernelPlot(3, 'sigmoid', .001)
	p.set_data(x, y)
	p.set_title('Iris Dataset | 3D PCA Kernel: Sigmoid Gamma: .001 | Target Data')
	p.show_plot()


def tsne_plot(x, y):
	# 2D
	p = TSNEPlot(2, 42)
	p.set_data(x, y)
	p.set_title('Iris Dataset | 2D t-SNE | Target Data')
	p.show_plot()

	# 3D
	p = TSNEPlot(3, 42)
	p.set_data(x, y)
	p.set_title('Iris Dataset | 3D t-SNE | Target Data')
	p.show_plot()


def isomap_plot(x, y):
	# 2D
	p = IsoMapPlot(2)
	p.set_data(x, y)
	p.set_title('Iris Dataset | 2D Isomap | Target Data')
	p.show_plot()

	# 3D
	p = IsoMapPlot(3)
	p.set_data(x, y)
	p.set_title('Iris Dataset | 3D Isomap | Target Data')
	p.show_plot()
"""
    PlotGenerator.py
    Created by Adam Kohl
    2.1.2020

    This script contains a series of classes vital to generating plots.
    Class BasePlot - houses vital attributes & methods needed for all plots
    Class LinePlot - inherits BasePlot & generates a line plot capable of multiple series
    Class ScatterPlot - inherits BasePlot & generates a scatter plot capable of multiple series
    Class HistogramPlot - inherits BasePlot & generates a basic histogram
    Class DimReductionPlot - inherits BasePlot & contains vital methods to plot 2D & 3D dimensionality reduction plots
    Class PCAPlot - inherits DimReductionPlot & generates a basic PCA plot
    Class PCAKernelPlot - inherits DimReductionPlot & generates a PCA plot using kernel types
                        kernel - linear, poly, sigmoid, rbf, cosine, precomputed
                        gamma - default is 1 / number of features | applicable for rbf, poly, sigmoid
    Class TSNEPlot - inherits DimReductionPlot & generates a tSNE plot
                    random state refers to the seed used by the random number generator
    Class IsoMapPlot - inherits DimReductionPlot & generates an IsoMap plot
"""
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap

# locals
from Configuration import *


class BasePlot:
    def __init__(self):
        self.save_dir = FIGURE_DIRECTORY
        self.filename = "%032x" % random.getrandbits(128)
        self.save = False
        self.show = False
        self.legend = False
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.label_pad = None
        self.fontsize = None
        self.x_label = ''
        self.y_label = ''

    def set_save_directory(self, dir_str):
        self.save_dir = dir_str

    def set_title(self, title_str=''):
        self.ax.set_title(title_str, fontsize=self.fontsize)

    def set_labels(self, x_str, y_str):
        self.x_label = x_str
        self.y_label = y_str
        self.ax.set_xlabel(x_str, labelpad=self.label_pad, fontsize=self.fontsize)
        self.ax.set_ylabel(y_str, labelpad=self.label_pad, fontsize=self.fontsize)

    def set_aiaa_specifications(self):
        self.label_pad = 14
        self.fontsize = 14

    def show_legend(self):
        self.legend = True

    def show_plot(self):
        if self.legend:
            plt.legend()
        plt.show()

    def save_figure(self, directory_str=None, filename_str=None, tight_layout=True):
        if directory_str:
            if filename_str:
                f_path = os.path.join(directory_str, filename_str + '.png')
            else:
                f_path = os.path.join(directory_str, self.filename + '.png')
        if not directory_str:
            if filename_str:
                f_path = os.path.join(self.save_dir, filename_str + '.png')
            else:
                f_path = os.path.join(self.save_dir, self.filename + '.png')
        if tight_layout:
            plt.tight_layout()
        plt.savefig(f_path, format='png')


class LinePlot(BasePlot):
    def __init__(self):
        BasePlot.__init__(self)

    def add_series(self, data, name=None, plt_color=None):
        self.ax.plot(data[0], data[1], label=name, color=plt_color)


class ScatterPlot(LinePlot):
    def __init__(self):
        LinePlot.__init__(self)

    def add_series(self, data, name=None, marker_size=2, marker_shape='o', plt_color=None):
        self.ax.scatter(data[0], data[1], s=marker_size, label=name, marker=marker_shape, color=plt_color)

    def add_series_line(self, data, name=None, plt_color=None):
        self.ax.plot(data[0], data[1], label=name, color=plt_color)


class HistogramPlot(BasePlot):
    def __init__(self, bins=10):
        BasePlot.__init__(self)
        self.bins = bins

    def set_data(self, data, hist_labels=None):
        plt.hist(data, bins=self.bins, label=hist_labels, histtype='bar', ec='black')


class DimReductionPlot(BasePlot):
    def __init__(self, n_components):
        BasePlot.__init__(self)
        self.n_components = n_components
        self.x_reduced = None

    def _set_plot_axis(self, x_reduced, y):
        if self.n_components == 2:
            self.ax.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y, cmap=plt.cm.hot, edgecolor='k')
        if self.n_components == 3:
            self.ax = Axes3D(self.fig)
            self.ax.scatter(x_reduced[:, 0], x_reduced[:, 1], x_reduced[:, 2], c=y, cmap=plt.cm.hot, edgecolor='k')


class PCAPlot(DimReductionPlot):
    def __init__(self, n_components):
        DimReductionPlot.__init__(self, n_components)

    def set_data(self, x, y=None):
        pca = PCA(self.n_components)
        self.x_reduced = pca.fit_transform(x)
        self._set_plot_axis(self.x_reduced, y)


class PCAKernelPlot(DimReductionPlot):
    def __init__(self, n_components, kernel, gamma):
        DimReductionPlot.__init__(self, n_components)
        self.kernel = kernel
        self.gamma = gamma

    def set_data(self, x, y=None):
        pca = KernelPCA(n_components=self.n_components, kernel=self.kernel, gamma=0.04)
        self.x_reduced = pca.fit_transform(x)
        self._set_plot_axis(self.x_reduced, y)


class TSNEPlot(DimReductionPlot):
    def __init__(self, n_components, random_state):
        DimReductionPlot.__init__(self, n_components)
        self.random_state = random_state

    def set_data(self, x, y=None):
        t_sne = TSNE(n_components=self.n_components, random_state=self.random_state)
        self.x_reduced = t_sne.fit_transform(x)
        self._set_plot_axis(self.x_reduced, y)


class IsoMapPlot(DimReductionPlot):
    def __init__(self, n_components):
        DimReductionPlot.__init__(self, n_components)

    def set_data(self, x, y=None):
        t_sne = Isomap(n_components=self.n_components)
        self.x_reduced = t_sne.fit_transform(x)
        self._set_plot_axis(self.x_reduced, y)

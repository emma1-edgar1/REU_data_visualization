from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
import numpy as np
from PlotGenerator import *
from Utilities import *

"""
    Alex Data
"""
[data, targets] = alex_data()

"""
    Breast Cancer Wisconsin (diagnostic) dataset
    30 numeric, predictive attributes
    Malignant (212), outliers
    Benign (357), inliers
"""
# [data, targets] = load_breast_cancer(True)

# Standard PCA model creation
model = PCA()
model.fit(data)

"""
    Determining the number of dimensions to preserve the data's variance
    95% is commonly used
"""
variance = 0.95

# Proportion of data's variance along the axis of each principal component
print(model.explained_variance_ratio_)
print("\n\n")

# Cumulative sum of variance distribution
cumsum = np.cumsum(model.explained_variance_ratio_)

plot_data = [np.arange(len(cumsum)), cumsum]
p = LinePlot()
p.add_series(plot_data)
p.show_plot()


# Return index of principal axis preserving desired variance
axis_index = np.argmax(cumsum >= 0.95)

# Number of principal components preserving variance
d = axis_index + 1
print("Number of determined principal axes %i" % d)

# --- Second method ---
model2 = PCA(n_components=0.95)
model2.fit(data)
print(model2.explained_variance_ratio_)
print(model2.components_)

#Getting started

# First GIT repo push
#Second repo push

#install skikit-learn
# conda install -n
#install seaborn

# modules
import os
import numpy as np
import pandas as pd
from sklearn import datasets
import tensorflow as tf

# locals
from Utilities import *
from PlotGenerator import *
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


f_path = os.path.join(DATA_DIRECTORY, 'Biosensor_Data.csv')
df = pd.read_csv(f_path)

# What's going on with the data?
print(df.describe)
print(df.dtypes)
names = list(df.columns.values)

for name in names:
	if df[name].dtype == object:
		df[name] = df[name].astype(str)
		df[name] = df[name].map(lambda x: x.rstrip('%'))
		df[name] = pd.to_numeric(df[name], errors='coerce') * .01
		# x = list(df[df[name].isnull()].index)
		# print(x)
	elif df[name].dtype != float:
		df[name] = df[name].astype(float)

df_clean = df.dropna(0)

# Normalize df dependent variables
df_targets, df_deps = df_split(df_clean, 'Power ')
df_deps = df_normalize(df_deps)


for dependent_var in list(df_targets.columns.values):
	p = PCAKernelPlot(2, 'rbf', .04)
	p.set_data(df_deps, y=df_targets[dependent_var])
	p.set_title('BioSensor - %s | 2D PCA Kernel: RBF | Gamma: .04' % dependent_var)
	p.save_figure()
	print('stop')

	p = PCAKernelPlot(3, 'rbf', .04)
	p.set_data(df_deps, y=df_targets[dependent_var])
	p.set_title('BioSensor - %s | 3D PCA Kernel: RBF | Gamma: .04' % dependent_var)
	p.show_plot()

	p = PCAKernelPlot(2, 'sigmoid', .001)
	p.set_data(df_deps, y=df_targets[dependent_var])
	p.set_title('BioSensor - %s | 2D PCA Kernel: Sigmoid |  Gamma: .001' % dependent_var)
	p.show_plot()

	p = PCAKernelPlot(3, 'sigmoid', .001)
	p.set_data(df_deps, y=df_targets[dependent_var])
	p.set_title('BioSensor - %s | 3D PCA Kernel: Sigmoid |  Gamma: .001' % dependent_var)
	p.show_plot()

	p = TSNEPlot(2, 42)
	p.set_data(df_deps, y=df_targets[dependent_var])
	p.set_title('BioSensor - %s | 2D t-SNE | Random State: 42' % dependent_var)
	p.show_plot()

	p = TSNEPlot(3, 42)
	p.set_data(df_deps, y=df_targets[dependent_var])
	p.set_title('BioSensor - %s | 3D t-SNE | Random State: 42' % dependent_var)
	p.show_plot()

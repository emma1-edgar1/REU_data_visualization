"""
	Utilities.py
	Created by Adam Kohl
	2.1.2020

	This file contains helper functions
	Please feel free to add methods
"""
from Configuration import *
import pandas as pd
import os


def alex_data():
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
	targets, df_deps = df_split(df_clean, 'Power ')
	data = df_normalize(df_deps)
	return [data, targets]

def df_normalize(df, exclude_indices=None):
	df_norm = (df - df.min()) / (df.max() - df.min())
	if exclude_indices is not None:
		for i in enumerate(exclude_indices):
			if isinstance(i[1], str):
				df_norm.loc[i[1]] = df.loc[i[1]]
			else:
				df_norm.iloc[:, i[1]] = df.iloc[i[1]]
	return df_norm


def df_remove_all_nan(df, by_row=True):
	if by_row:
		return df.dropna(0)
	else:
		return df.dropna(1)


def df_split(df, index):
	if isinstance(index, int):
		return df.iloc[:, :index], df.iloc[:, index:]
	elif isinstance(index, str):
		is_found = False
		for (i, name) in enumerate(df.columns.tolist()):
			if name == index:
				is_found = True
				i += 1
				break
		if is_found:
			return df.iloc[:, :i], df.iloc[:, i:]
		else:
			print(f"COLUMN NOT FOUND USING '{index}'")
			return None, None
	else:
		print("NEED AN INT OR STR TO SPLIT DF")
		return None, None

import scipy.io
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(23, input_dim=23, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def import_mat(folder_path, file_list):
	input_model = []
	output_model_CBF = []
	for file in file_list:
		mat = scipy.io.loadmat(folder_path + "/" + file)
		# print(mat['X'].shape)
		# for k in mat.keys():
		# 	if not k.startswith('__'):
		# 		print(k + " " + mat[k].dtype.name + " " + str(mat[k].shape))
		input_model, output_model_CBF = build_input_output(mat, input_model, output_model_CBF)
		# label = mat['CBF'][0][0]
		# print(label)
		# for i in range(20):
		# 	# The greyscale of picture are not on the same level
		# 	# Do we need to standardize them to the same level?
		# 	img = mat['X'][:, :, i]
		# 	plt.figure()
		# 	plt.imshow(img, cmap='gray')
		# 	plt.show()
		# print(len(inputs))
	# fix random seed for reproducibility
	seed = 7
	numpy.random.seed(seed)
	# evaluate model with standardized dataset
	estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
	kfold = KFold(n_splits=10, random_state=seed)
	results = cross_val_score(estimator, input_model, output_model_CBF, cv=kfold)
	print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

def build_input_output(mat, input_model, output_model):
	h, w, n = mat['X'].shape
	for i in range(h):
		for j in range(w):
			input_pixel = extract_pixels(mat['X'], i, j)
			print(input_pixel)
			input_model.append(input_pixel)
			output_model.append(mat['CBF'][i][j])
	return input_model, output_model

def extract_pixels(images, h_cord, w_cord):
	_, _, n = images.shape
	input_array = []
	for i in range(n):
		input_array.append(images[h_cord][w_cord][i])
	input_array.append(h_cord)
	input_array.append(w_cord)
	return input_array

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python3 process.py [path of data folder]")
		exit()
	folder_path = sys.argv[1]
	data_list = os.listdir(folder_path)
	import_mat(folder_path, data_list)




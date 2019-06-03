import scipy.io
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn.kernel_ridge import KernelRidge
from sklearn import model_selection
import pickle


def import_mat(folder_path, file_list):
	#print(file_list)
	for file in file_list:
		file_path = folder_path + "/" + file
		#print(file)
		mat = scipy.io.loadmat(file_path)
		# print(mat['X'].shape)
		# for k in mat.keys():
		# 	if not k.startswith('__'):
		# 		print(k + " " + mat[k].dtype.name + " " + str(mat[k].shape))
		# inputs = extract_pixels(mat['X'], 0, 0)
		#print(inputs)
		# label = mat['CBF'][0][0]
		label = mat['CBV']
		# flatten = mat['CBV'].flatten()
		# max_val = np.amin(flatten)
		
		
		for i in range(20):
			# The greyscale of picture are not on the same level
			# Do we need to standardize them to the same level? normalization.
			img = mat['X'][:, :, i]
			# img = tf.keras.utils.normalize(img,axis =1)
			# print(img)
			# plt.figure()
			# plt.imshow(img,cmap = plt.cm.binary)
			# plt.show()
		# print(len(inputs))

def import_data(inp, outp):
	input_array = np.asarray(inp, dtype=np.float32)
	output_array = np.asarray(outp, dtype=np.float32)
	# print("Input array shape: ")
	# print(input_array.shape)
	# print("Output array shape: ")
	# print(output_array.shape)
	krr_model = build_model(input_array, output_array)
	performance_measure(krr_model, input_array, output_array)

def build_model(input_train, label_train):
	#!!!May need to change the kernel function and alpha value
	kr = KernelRidge(kernel="poly", degree=6, alpha=1).fit(input_train, label_train)
	return kr

def predict_result(model, input_data):
	return model.predict(input_data)

def performance_measure(model, input_data, output_data):
	n = len(input_data)
	MSEs_KRR = model_selection.cross_val_score(model, input_data, output_data, cv=model_selection.LeavePOut(p=1), scoring="neg_mean_squared_error")
	MeanMSE_KRR = np.mean(list(MSEs_KRR))
	print("MSE: " + MeanMSE_KRR)

def listdir_nohidden(path):
	for f in os.listdir(path):
		if not f.startswith('.'):
			yield f

def extract_pixels(images, h_cord, w_cord):
	_, _, n = images.shape
	input_array = []
	for i in range(n):
		input_array.append(images[h_cord][w_cord][i])
	# input_array.append(h_cord)
	# input_array.append(w_cord)
	return input_array

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("Usage: python3 process.py [input_file_path] [output_file_path]")
		exit()
	input_file_path = sys.argv[1]
	output_file_path = sys.argv[2]
	with open(input_file_path,'rb') as f:	
		inp = pickle.load(f)
	with open(output_file_path,'rb') as v:
		outp = pickle.load(v)
	import_data(inp, outp)




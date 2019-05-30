import scipy.io
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


def import_mat(folder_path, file_list):
	for file in file_list:
		file_path = folder_path + "/" + file
		print(file)
		mat = scipy.io.loadmat(file_path)
		
		# print(mat['X'].shape)
		# for k in mat.keys():
		# 	if not k.startswith('__'):
		# 		print(k + " " + mat[k].dtype.name + " " + str(mat[k].shape))
		inputs = extract_pixels(mat['X'], 0, 0)
		print(inputs)
		label = mat['CBF'][0][0]
		#print(label)
		for i in range(20):
			# The greyscale of picture are not on the same level
			# Do we need to standardize them to the same level?  normalization.
			img = mat['X'][:, :, i]
			img = tf.keras.utils.normalize(img,axis =1)
			print(img)
			plt.figure()
			plt.imshow(img,cmap = plt.cm.binary)
			plt.show()
		# print(len(inputs))

def listdir_nohidden(path):
	for f in os.listdir(path):
		if not f.startswith('.'):
			yield f

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
	folder_path = os.path.abspath(folder_path)
	data_list = listdir_nohidden(folder_path)
	print(data_list)
	import_mat(folder_path, data_list)




import scipy.io
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def import_mat(folder_path, file_list):
	for file in file_list:
		mat = scipy.io.loadmat(folder_path + "/" + file)
		# print(mat['X'].shape)
		# for k in mat.keys():
		# 	if not k.startswith('__'):
		# 		print(k + " " + mat[k].dtype.name + " " + str(mat[k].shape))
		inputs = extract_pixels(mat['X'], 0, 0)
		print(inputs)
		label = mat['CBF'][0][0]
		print(label)
		for i in range(20):
			# The greyscale of picture are not on the same level
			# Do we need to standardize them to the same level?
			img = mat['X'][:, :, i]
			plt.figure()
			plt.imshow(img, cmap='gray')
			plt.show()
		# print(len(inputs))

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




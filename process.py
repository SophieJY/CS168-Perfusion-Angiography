import scipy.io
import sys
import os
import numpy as np

def import_mat(folder_path, filename):
	mat = scipy.io.loadmat(folder_path + "/" + filename)
	# print(mat['X'].shape)
	# print(mat['X'][0][0][1])
	for k in mat.keys():
		if not k.startswith('__'):
			print(k + " " + mat[k].dtype.name + " " + str(mat[k].shape))
	inputs = extract_pixels(mat['X'], 0, 0)
	print(len(inputs))

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
	for data in data_list:
		import_mat(folder_path, data)




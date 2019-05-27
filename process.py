import scipy.io
import sys
import os
import numpy as np

def import_mat(folder_path, filename):
	mat = scipy.io.loadmat(folder_path + "/" + filename)
	print(mat['X'].shape)
	# print(mat['X'].keys())
	for k in mat.keys():
		if not k.startswith('__'):
			print(k + " " + mat[k].dtype.name + " " + str(mat[k].shape))

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python3 process.py [path of data folder]")
	folder_path = sys.argv[1]
	data_list = os.listdir(folder_path)
	for data in data_list:
		import_mat(folder_path, data)




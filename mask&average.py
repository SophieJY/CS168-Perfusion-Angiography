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
from sklearn.metrics import mean_squared_error

average_list = {}

def import_mat(folder_path, file_list):
	#print(file_list)
	for file in file_list:
		name = file.split('.')[0]
		patient_id = name.split('_')[-1]
		if patient_id[0] == '0':
			patient_id = patient_id[1:]
		averages = []
		file_path = folder_path + "/" + file
		mat = scipy.io.loadmat(file_path)
		# for k in mat.keys():
		# 	if not k.startswith('__'):
		# 		print(k + " " + mat[k].dtype.name + " " + str(mat[k].shape))
		roi = mat['roi']
		h, w = roi.shape
		# print(roi)
		# plt.figure()
		# plt.imshow(roi,cmap = plt.cm.binary)
		# plt.show()
		mask = []
		for i in range(h):
			for j in range(w):
				if roi[i][j] == 1:
					mask.append((i, j))
		# print(mask)
		mask_len = len(mask)
		# print("length of mask: " + str(mask_len))
		for i in range(20):
			img = mat['X'][:, :, i]
			sum = 0
			for coordinate in mask:
				sum += img[coordinate[0], coordinate[1]]
			average = sum/mask_len
			averages.append(average)
		# print(averages)
		print(len(averages))
		average_list[patient_id] = averages
	print(len(average_list))
	print(average_list)

	with open('averagelist.pkl','wb') as f:
		pickle.dump(average_list,f,protocol=pickle.HIGHEST_PROTOCOL)

def listdir_nohidden(path):
	for f in os.listdir(path):
		if not f.startswith('.'):
			yield f

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python3 process.py [path of data folder]")
		exit()
	folder_path = sys.argv[1]
	folder_path = os.path.abspath(folder_path)
	# print(os.listdir(folder_path))
	data_list = listdir_nohidden(folder_path)
	#print(data_list)
	import_mat(folder_path, data_list)

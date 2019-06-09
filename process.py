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
import matplotlib as mpl
from matplotlib import cm

clean_data_num = [6, 23, 24, 31, 32, 41, 42, 52, 53, 54, 75, 106]
# train_data_num = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 26, 27, 29, 30, 34, 35, 36, 37, 38, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 80, 81, 82, 83, 84, 85, 87, 89, 90, 91, 92, 93, 94, 95]

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

def show_color_map(img):
	print("colormap")
	cmap = mpl.cm.jet
	cmap_r = cmap
	plt.imshow(img,cmap=cmap_r)
	plt.show()

def import_data(input_train_folder, output_train_folder, input_test_folder, output_test_folder):
	X_train = put_together_extended(input_train_folder)
	y_train = put_together(output_train_folder)
	iter_num = get_length(input_test_folder)
	X_test_files = os.listdir(input_test_folder)
	y_test_files = os.listdir(output_test_folder)
	X_test_files.sort()
	y_test_files.sort()
	print(X_test_files)
	print(y_test_files)
	# y_test = form_matrix(output_test_folder, y_test_files[0])
	# original_image = np.reshape(y_test, (1024, 1024))
	# print(original_image[0][0])
	# original_img = Image.fromarray(original_image, 'RGB')
	# original_img.save('output_orig.png')
	# original_img.show()
	# show_color_map(original_image)

	# y_test = put_together(output_test_folder)
	# input_array = np.asarray(inp, dtype=np.float32)
	# output_array = np.asarray(outp, dtype=np.float32)
	# predict_array = np.asarray(predp, dtype=np.float32)
	# print("Input array shape: ")
	# print(input_array.shape)
	# print("Output array shape: ")
	# print(output_array.shape)
	# X_train, X_test, y_train, y_test = model_selection.train_test_split(input_array, output_array, test_size=0.15, random_state=i)
	krr_model = build_model(X_train, y_train)
	mse_sum = 0
	for i in range(iter_num):
		X_test = form_matrix(input_test_folder, X_test_files[i])
		y_predict = np.zeros((1024, 1024))
		count = 0
		for line in X_test:
			# print(line)
			element = predict_result(krr_model, line.reshape(1, -1))
			# print(element)
			y_predict[count//1024][count%1024] = element[0]
			print(count)
			count+=1
		# y_predict = predict_result(krr_model, X_test)
		image = np.reshape(y_predict, (1024, 1024))
		img = Image.fromarray(image, 'L')
		img.save('output_new.png')
		img.show()
		show_color_map(image)
		y_test = form_matrix(output_test_folder, y_test_files[i])
		original_image = np.reshape(y_test, (1024, 1024))
		original_img = Image.fromarray(original_image, 'L')
		original_img.save('output_orig.png')
		original_img.show()
		show_color_map(original_image)
		print(y_predict.shape)
		print(y_test)
		print(y_predict.flatten())
		print(original_image[100][100])
		print(image[100][100])
		mse_sum += performance_measure(krr_model, y_predict.flatten(), y_test)
	print("Mse: ")
	print(mse_sum/iter_num)

def put_together_extended(folder_path):
	input_matrix = []
	data_list = get_list(folder_path)
	threshold = 40
	for file in data_list:
		if file == '_DS_Store':
			continue
		name = file.split('.')[0]
		patient_id = name.split('_')[-1]
		if patient_id not in clean_data_num:
			with open(folder_path + "/" + file,'rb') as f:	
				inp = pickle.load(f)
				# print("input length: ")
				# print(len(inp))
				# print("matric length: ")
				# print(len(input_matrix))
				input_matrix = input_matrix + inp.tolist()
				# print("matric length: ")
				# print(len(input_matrix))
	train_matrix = np.asarray(input_matrix, dtype=np.float32)
	print("Input Matrix shape: ")
	print(train_matrix.shape)
	return train_matrix

def put_together(folder_path):
	input_matrix = []
	data_list = get_list(folder_path)
	threshold = 40
	for file in data_list:
		if file == '_DS_Store':
			continue
		name = file.split('.')[0]
		patient_id = name.split('_')[-1]
		if patient_id not in clean_data_num:
			with open(folder_path + "/" + file,'rb') as f:	
				inp = pickle.load(f)
				# print("input length: ")
				# print(len(inp))
				# print("matric length: ")
				# print(len(input_matrix))
				input_matrix = input_matrix + inp
				# print("matric length: ")
				# print(len(input_matrix))
	train_matrix = np.asarray(input_matrix, dtype=np.float32)
	print("Input Matrix shape: ")
	print(train_matrix.shape)
	return train_matrix

def get_length(test_folder):
	data_list = os.listdir(test_folder)
	# print(len(data_list))
	# print(len(clean_data_num))
	return len(data_list)

def form_matrix(folder_path, file):
	with open(folder_path + "/" + file,'rb') as f:	
		print(file)
		inp = pickle.load(f)
		input_array = np.asarray(inp, dtype=np.float32)
		print("test_input_array_size:")
		print(input_array.shape)
	return input_array

def build_model(input_train, label_train):
	#!!!May need to change the kernel function and alpha value
	kr = KernelRidge(kernel="poly", degree=5, alpha=1).fit(input_train, label_train)
	return kr

def predict_result(model, input_data):
	return model.predict(input_data)

def performance_measure(model, predicted_result, label_test):
	mse = mean_squared_error(label_test, predicted_result)
	print("MSE: ")
	print(mse)
	# print("Score: ")
	# print(model.score(input_test, label_test))
	return mse
	# n = len(input_data)
	# MSEs_KRR = model_selection.cross_val_score(model, input_data, output_data, cv=model_selection.LeavePOut(p=1), scoring="neg_mean_squared_error")
	# MeanMSE_KRR = np.mean(list(MSEs_KRR))
	# print("MSE: " + MeanMSE_KRR)

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

def get_list(folder_path):
	folder_path = os.path.abspath(folder_path)
	# print(os.listdir(folder_path))
	data_list = listdir_nohidden(folder_path)
	return data_list

if __name__ == "__main__":
	if len(sys.argv) < 5:
		print("Usage: python3 process.py [input_train_folder] [output_train_folder] [input_test_folder] [output_test_folder]")
		exit()
	input_train_folder = sys.argv[1]
	output_train_folder = sys.argv[2]
	input_test_folder = sys.argv[3]
	output_test_folder = sys.argv[4]
	# with open(input_file_path,'rb') as f:	
	# 	inp = pickle.load(f)
	# with open(output_file_path,'rb') as v:
	# 	outp = pickle.load(v)
	# with open(predict_file_path,'rb') as p:
	# 	predp = pickle.load(p)
	import_data(input_train_folder, output_train_folder, input_test_folder, output_test_folder)




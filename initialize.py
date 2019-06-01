import scipy.io
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from PIL import Image
import tensorflow as tf
from process import listdir_nohidden,import_mat,extract_pixels
import pickle

#global vars
CBV = []
X = []
x_train = []
y_train = []

def preprocessing(folder_path):
	for i in range(1,132):
		file_path = folder_path + "/" + "perfAngio_" + str(i)
		mat = scipy.io.loadmat(file_path)
		label = mat['CBV']
		# flatten_cbv = label.flatten()
		CBV.append(label)
		x = mat['X']
		X.append(x)
		print(i)
		# flatten_x = []
		# for i in range(20):
		# 	img = mat['X'][:, :, i]
		# 	img = img.flatten()
		# 	flatten_x.append(img)
		# flatten_X.append(flatten_x)
	# file_path = folder_path + "/" + "perfAngio_" + str(105)
	# mat = scipy.io.loadmat(file_path)
	# label = mat['CBV']
	# flatten_cbv = label.flatten()
	# flatten_CBV.append(flatten_cbv)
	# flatten_x = []
	# for i in range(20):
	# 	img = mat['X'][:, :, i]
	# 	img = img.flatten()
	# 	flatten_x.append(img)
	# flatten_X.append(flatten_x)

def bucket_sampling():
	full = False
	full_0 = False #4.440892098500626e-15
	full_1 = False #4.440892098500626e-15 ~ 1
	full_2 = False #1 ~ 5
	full_3 = False #5 ~ 10
	full_4 = False #10 ~ 15
	full_5 = False #15 ~ 20
	full_6 = False #20 ~ 30
	full_7 = False #30 ~ 40
	full_8 = False #40 ~ 50
	full_9 = False #50 ~ 80
	bk_0_num = 0
	bk_1_num = 0
	bk_2_num = 0
	bk_3_num = 0
	bk_4_num = 0
	bk_5_num = 0
	bk_6_num = 0
	bk_7_num = 0
	bk_8_num = 0
	bk_9_num = 0
	threshold = 5000
	while full == False:
		index = randint(0,130)
		row_num = randint(0,1023)
		col_num = randint(0,1023)		
		y = CBV[index][row_num][col_num]
		# print(y)
		if y == 4.440892098500626e-15:
			if bk_0_num < threshold:
				bk_0_num += 1
				# print("a")
				# print(bk_0_num)
				x = extract_pixels(X[index],row_num,col_num)
				x_train.append(x)
				y_train.append(y)
			if bk_0_num == threshold:
				full_0 = True
		elif y > 4.440892098500626e-15 and y <= 1:
			if bk_1_num < threshold:
				bk_1_num += 1
				# print("b")
				# print(bk_1_num)
				x = extract_pixels(X[index],row_num,col_num)
				x_train.append(x)
				y_train.append(y)
			if bk_1_num == threshold:
				full_1 = True
		elif y > 1 and y <= 5:
			if bk_2_num < threshold:
				bk_2_num += 1
				# print("c")
				# print(bk_2_num)
				x = extract_pixels(X[index],row_num,col_num)
				x_train.append(x)
				y_train.append(y)
			if bk_2_num == threshold:
				full_2 = True
		elif y > 5 and y <= 10:
			if bk_3_num < threshold:
				bk_3_num += 1
				# print("d")
				# print(bk_3_num)
				x = extract_pixels(X[index],row_num,col_num)
				x_train.append(x)
				y_train.append(y)
			if bk_3_num == threshold:
				full_3 = True
		elif y > 10 and y <= 15:
			if bk_4_num < threshold:
				bk_4_num += 1
				# print("e")
				# print(bk_4_num)
				x = extract_pixels(X[index],row_num,col_num)
				x_train.append(x)
				y_train.append(y)
			if bk_4_num == threshold:
				full_4 = True
		elif y > 15 and y <= 20:
			if bk_5_num < threshold:
				bk_5_num += 1
				# print("f")
				# print(bk_5_num)
				x = extract_pixels(X[index],row_num,col_num)
				x_train.append(x)
				y_train.append(y)
			if bk_5_num == threshold:
				full_5 = True
		elif y > 20 and y <= 30:
			if bk_6_num < threshold:
				bk_6_num += 1
				x = extract_pixels(X[index],row_num,col_num)
				# if x[0] == 0.0:
				# 	print("----")
				# 	print(y)
				# 	print(index)
				# 	print(row_num)
				# 	print(col_num)
				x_train.append(x)
				y_train.append(y)
			if bk_6_num == threshold:
				full_6 = True
		elif y > 30 and y <= 40:
			if bk_7_num < threshold:
				bk_7_num += 1
				x = extract_pixels(X[index],row_num,col_num)
				# if x[0] == 0.0:
				# 	print("----")
				# 	print(y)
				# 	print(index)
				# 	print(row_num)
				# 	print(col_num)
				x_train.append(x)
				y_train.append(y)
			if bk_7_num == threshold:
				full_7 = True
		elif y > 40 and y <= 50:
			if bk_8_num < threshold:
				bk_8_num += 1
				x = extract_pixels(X[index],row_num,col_num)
				# if x[0] == 0.0:
				# 	print("----")
				# 	print(y)
				# 	print(index)
				# 	print(row_num)
				# 	print(col_num)
				x_train.append(x)
				y_train.append(y)
			if bk_8_num == threshold:
				full_8 = True
		elif y > 50 and y <= 80:
			if bk_9_num < threshold:
				bk_9_num += 1
				x = extract_pixels(X[index],row_num,col_num)
				# if x[0] == 0.0:
				# 	print("----")
				# 	print(y)
				# 	print(index)
				# 	print(row_num)
				# 	print(col_num)
				x_train.append(x)
				y_train.append(y)
			if bk_9_num == threshold:
				full_9 = True
		if(full_0 == True and full_1 == True and full_2 == True and full_3 == True and full_4 == True and full_5 == True and full_6 == True and full_7 == True and full_8 == True and full_9 == True):
			# print("x")
			full = True
			break





if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python3 process.py [path of data folder]")
		exit()
	folder_path = sys.argv[1]
	folder_path = os.path.abspath(folder_path)
	# print(os.listdir(folder_path))
	data_list = os.listdir(folder_path)
	#print(data_list)
	data_list.pop(0)
	preprocessing(folder_path)
	# print(CBV)
	print("preprocessing complete")
	# print(X)
	# flatten_CBV = np.array(flatten_CBV)
	bucket_sampling()
	with open('input.pkl','wb') as f:
		pickle.dump(x_train,f)
	with open('output.pkl','wb') as v:
		pickle.dump(y_train,v)

	# with open('input.pkl','rb') as f:	
	# 	inp = pickle.load(f)
	# with open('output.pkl','rb') as v:
	# 	outp = pickle.load(v)
	# print(inp[40000:41000])
	# print(outp[40000:41000])
	# print(np.array(X))
	# print(X)
	# print(x_train)
	# print(y_train)
		# print(flatten_CBV[0])
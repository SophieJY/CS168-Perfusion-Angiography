import scipy.io
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle

def process(folder_path, data_list, average_list_file):
	with open(average_list_file,'rb') as f:	
		average_dict = pickle.load(f)
		print(type(average_dict))
	for data_file in data_list:
		name = data_file.split('.')[0]
		patient_id = name.split('_')[-1]
		new_input_array = []
		with open(folder_path + "/" + data_file,'rb') as d:
			inp = pickle.load(d)
			input_array = np.asarray(inp, dtype=np.float32)
			print(input_array.shape)
			vector_to_append = average_dict[patient_id]
			# print(vector_to_append)
			for vector in input_array:
				new_vector = vector_to_append + vector.tolist()
				new_input_array.append(new_vector)
		new_input = np.asarray(new_input_array, dtype=np.float32)
		print(new_input.shape)
		with open(folder_path + "_extended" +"/" + data_file,'wb') as h:
			pickle.dump(new_input,h)
		# with open(folder_path + "_extended" +"/" + data_file,'rb') as k:
		# 	inp = pickle.load(k)
		# 	input_array = np.asarray(inp, dtype=np.float32)
		# 	print(input_array.shape)

def listdir_nohidden(path):
	for f in os.listdir(path):
		if not f.startswith('.'):
			if f.startswith('input'):
				yield f

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("Usage: python3 append.py [input_file_folder] [average_list_file]")
		exit()
	input_file_folder = sys.argv[1]
	average_list_file = sys.argv[2]
	folder_path = os.path.abspath(input_file_folder)
	# print(os.listdir(folder_path))
	data_list = listdir_nohidden(folder_path)
	process(folder_path, data_list, average_list_file)
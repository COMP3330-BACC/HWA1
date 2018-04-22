import os 						 # os 			-- OS operations, read, write etc
import tensorflow as tf 		 # tensorflow 	-- machine learning
import numpy as np 				 # numpy		-- python array operations
import matplotlib.pyplot as plt  # matplotlib 	-- plotting
import time						 # time 		-- performance measure
import csv
import random
import string
from sklearn.svm import SVC

import sys
sys.path.insert(0, '../../')
import util 

def load_data(data_file):
	with open(data_file, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		x = []
		y = []

		for row in csvreader:
			x_conv = (list(map(float, row[:-1])))
			y_conv = ([float(row[-1])])
			x.append(x_conv)
			y.append(y_conv)
			#print(x_conv)
			#print(y_conv)
		#y=relabel(y)
		x, y = util.shuffle_data(x, y)
		x_train, x_test, y_train, y_test = util.split_data(x, y, 0.9)

		return x_train, x_test, y_train, y_test
	print('[ERR] Failed to load data from file \'{0}\''.format(data_file))
	exit()
'''
def relabel(data):
	new_data = []
	classes = 10
	for row in data:
		new_row = [0] * classes
		new_row[int(row[0])] = 1
		new_data.append(new_row)
	return new_data
'''
def train_svm (x, y, cfg):
	C      = cfg['svm']['C']
	kernel = cfg['svm']['kernel']
	gamma  = cfg['svm']['gamma']

	print('\t[SVM] Training parameters: C={0}, kernel={1}, gamma={2}'.format(C, kernel, gamma))

	svm = SVC(C=C, kernel=kernel, gamma=gamma)

	t_start=time.time()
	
#	print((np.array(x)).shape)
#	print((np.ravel(y)).shape)

	svm.fit(np.array(x), np.ravel(y))
	t_train = time.time()-t_start
	print('\t[SVM] Model accuracy: {0:.2f}%, Time to train: {1:.5f}s'.format(100*svm.score(x, np.ravel(y)), t_train))
	
	return svm

def test_svm(svm, x):
	t_start = time.time()
	y = [svm.predict([sample]) for sample in x]
	t_avg_test = (time.time() - t_start)/float(len(y))

	print('\t[SVM] Average time to test: {0:.2f}us'.format(1000000 * t_avg_test))
	return y

def returnNotMatches(a, b):
    return [[x for x in a if x not in b], [x for x in b if x not in a]]

def main():
	cfg = util.read_config('config/digit.yaml')

	x_train, x_test, y_train, y_test = load_data(cfg['dataset'])
	x_train, y_train = util.shuffle_data(x_train, y_train)
	x_test, y_test   = util.shuffle_data(x_test, y_test)

	model_name = cfg['svm']['model_name']
	model_dir = cfg['svm']['model_dir']

	svm = train_svm(x_train, y_train, cfg)
	y_test2 = test_svm(svm, x_test)

	print(returnNotMatches(y_test, y_test2))

if __name__ == '__main__':
	main()	
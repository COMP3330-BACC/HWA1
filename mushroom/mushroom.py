import os 									# os 			-- OS operations, read, write etc
import tensorflow as tf 					# tensorflow 	-- machine learning
import numpy as np 							# numpy			-- python array operations
import matplotlib.pyplot as plt 			# matplotlib 	-- plotting
import csv									# csv 			-- reading from CSVs easily
import yaml									# yaml 			-- reading/writing config files

# Will load data in form of
#	data = [(x_0, y_0, class_0) ... (x_n, y_n, class_n)]
def load_data(data_file):
	with open(data_file, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		data = [row for row in csvreader]
		return data
	print('[ERR] Failed to load data from file \'{0}\''.format(data_file))
	exit()

def read_config(cfg_file='config/mushroom.yaml'):
	with open(cfg_file, 'r') as yml:
		return yaml.load(yml)
	print('[ERR] Failed to load config file \'{0}\''.format(cfg_file))
	exit()

## Main Program

# Read config file
cfg  = read_config()

# Load two spiral data from dataset
ts_data = load_data(cfg['mushroom']['dataset'])

print('{0} ... {1}'.format(ts_data[0], ts_data[-1]))
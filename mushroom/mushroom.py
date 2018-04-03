import os 						 # os 			-- OS operations, read, write etc
import tensorflow as tf 		 # tensorflow 	-- machine learning
import numpy as np 				 # numpy		-- python array operations
import matplotlib.pyplot as plt  # matplotlib 	-- plotting
import yaml						 # yaml 		-- reading/writing config files
import time						 # time 		-- performance measure
import csv
import pandas as pd
import pickle
import random

# Shuffle data to randomize order of samples
def shuffle_data(x, y):
	combined = list(zip(x, y))
	random.shuffle(combined)
	return zip(*combined)

# Split data into training and testing datasets
def split_data(x, y, train_ratio=0.8):
	# Create pivot point within dataset
	pivot = int(train_ratio * len(x))
	return x[:pivot], x[pivot:], y[:pivot], y[pivot:]

# Will load data in form as specified by the dataset description 
#	(https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names) 
# Will load data in from the spiral dataset csv as store as x = [(x, y) ...] and y = [(c) ...]
def load_data(data_file):
	with open(data_file, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		x = []
		y = []

		for row in csvreader:
			x_conv = list(map(float, map(ord, row[1:])))
			y_conv = [0. if row[0] == 'p' else 1.]
			# Remove question marks
			for u in x_conv:
				if u == 63.:
					u = -1.
			x.append(x_conv)
			y.append(y_conv)
			# print(x_conv)
			# print(y_conv)

		x, y = shuffle_data(x, y)
		x_train, x_test, y_train, y_test = split_data(x, y, 0.9)

		return x_train, x_test, y_train, y_test
	print('[ERR] Failed to load data from file \'{0}\''.format(data_file))
	exit()

def read_config(cfg_file='config/mushroom.yaml'):
	with open(cfg_file, 'r') as yml:
		return yaml.load(yml)
	print('[ERR] Failed to load config file \'{0}\''.format(cfg_file))
	exit()

def construct_network(inp, weights, biases, neurons):
	fc1 = tf.nn.sigmoid(tf.add((tf.matmul(inp, weights['fc1'])), biases['fc1']), name='fc1')
	fc2 = tf.nn.sigmoid(tf.add((tf.matmul(fc1, weights['fc2'])), biases['fc2']), name='fc2')
	fc3 = tf.nn.sigmoid(tf.add((tf.matmul(fc2, weights['fc3'])), biases['fc3']), name='fc3')
	fc4 = tf.nn.sigmoid(tf.add((tf.matmul(fc3, weights['fc4'])), biases['fc4']), name='fc4')
	fc5 = tf.nn.sigmoid(tf.add((tf.matmul(fc4, weights['fc5'])), biases['fc5']), name='fc5')
	fc6 = tf.nn.sigmoid(tf.add((tf.matmul(fc5, weights['fc6'])), biases['fc6']), name='fc6')

	return fc6

def save_model(sess, weights, biases, neurons, path):
	with open(path, "wb") as pf:
		# Create new dictionary containers for non-tf types
		n_weights = {}; n_biases = {}
		for key in weights.keys():
			n_weights.update({key : sess.run(weights[key])})
		for key in biases.keys():
			n_biases.update({key : sess.run(biases[key])})
		model = {'weights' : n_weights, 'biases' : n_biases, 'neurons' : neurons}
		pickle.dump(model, pf)
		return model

def load_model(path):
	with open(path, "rb") as pf:
		model = pickle.load(pf)
		return model

def train_network(sess, x, y, cfg):
	# Alias config vars
	neurons       = cfg['training']['nn']['neurons']
	epochs        = cfg['training']['nn']['epochs']
	learning_rate = cfg['training']['nn']['learning_rate']
	iterations    = cfg['training']['nn']['iterations']
	acc_thresh    = cfg['training']['nn']['accuracy_threshold']
	model_dir     = cfg['training']['nn']['model_dir']

	print('[ANN] \tTraining parameters: epochs={0}, learning_rate={1:.2f}, neurons={2}'.format(epochs, learning_rate, neurons))

	# Create validation set
	x_train, x_valid, y_train, y_valid = split_data(x, y, 0.9)
	x_valid, y_valid = shuffle_data(x_valid, y_valid)

	# Create placeholders for tensors
	x_ = tf.placeholder(tf.float32, [None, 22], name='x_placeholder')
	y_ = tf.placeholder(tf.float32, [None, 1],  name='y_placeholder')

	weights = {
		'fc1' : tf.Variable(tf.random_normal([22, neurons]), name='w_fc1'),
		'fc2' : tf.Variable(tf.random_normal([neurons, neurons]), name='w_fc2'),
		'fc3' : tf.Variable(tf.random_normal([neurons, neurons]), name='w_fc3'),
		'fc4' : tf.Variable(tf.random_normal([neurons, neurons]), name='w_fc4'),
		'fc5' : tf.Variable(tf.random_normal([neurons, neurons]), name='w_fc5'),
		'fc6' : tf.Variable(tf.random_normal([neurons, 1]), name='w_fc6'),
	}

	biases = {
		'fc1' : tf.Variable(tf.random_normal([neurons]), name='b_fc1'),
		'fc2' : tf.Variable(tf.random_normal([neurons]), name='b_fc2'),
		'fc3' : tf.Variable(tf.random_normal([neurons]), name='b_fc3'),
		'fc4' : tf.Variable(tf.random_normal([neurons]), name='b_fc4'),
		'fc5' : tf.Variable(tf.random_normal([neurons]), name='b_fc5'),
		'fc6' : tf.Variable(tf.random_normal([1]), name='b_fc6'),
	}

	final_layer = construct_network(x_, weights, biases, neurons)

	# Define error function
	cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=final_layer))

	# Define optimiser and minimise error function task
	optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

	# Create error logging storage
	train_errors = []
	valid_errors = []

	# Initialise global variables of the session
	sess.run(tf.global_variables_initializer())

	# Save model to file
	model = save_model(sess, weights, biases, neurons, model_dir)

	# Setup our continous plot
	plt.title('Error vs Epoch')
	plt.plot(train_errors[:epochs], color='r', label='training')
	plt.plot(valid_errors[:epochs], color='b', label='validation')
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.legend()
	plt.grid()
	plt.ion()
	plt.show()

	# Measure training time
	t_start = time.time()

	for i in range(epochs):
		_, train_error = sess.run([optimiser, cost], feed_dict={x_: x_train, y_: y_train})
		_, valid_error = sess.run([optimiser, cost], feed_dict={x_: x_valid, y_: y_valid})
		print('{0:.4f}'.format(valid_error - train_error))
		train_errors.append(train_error)
		valid_errors.append(valid_error)
		if train_error <= 1 - acc_thresh:
			break
		
		# Set plot settings
		plt.plot(train_errors[:epochs], color='r', label='training')
		plt.plot(valid_errors[:epochs], color='b', label='validation')
		plt.draw()
		plt.pause(0.001)			

	t_elapsed = time.time() - t_start

	# Calculate new accuracy
	accuracy = 1 - error

	print('[ANN] \tModel accuracy: {0:.3f}%, Time elapsed: {1:.2f}s'.format(accuracy*100, t_elapsed))
	return model

def test_network(sess, model, x_test, y_test, cfg):
	x_t = tf.placeholder(tf.float32, [None, 22], name='t_x_placeholder')
	y_t = tf.placeholder(tf.float32, [None, 1],  name='t_y_placeholder')

	final_layer = construct_network(x_t, model['weights'], model['biases'], model['neurons'])

	# Start timing the length of time training takes
	t_test = time.time()

	# Classify test data
	result = np.round(sess.run(final_layer, feed_dict={x_t : x_test, y_t : y_test}))

	# Average out the test timing
	t_avg_test = (time.time() - t_test) / float(len(y_test))

	print('[ANN] Average time to test: {0:.2f}us'.format(1000000 * t_avg_test))
	return result

def analyse_results(y_test, results):
	# Make variables for true pos, true neg, false pos, false neg
	tp = 0; tn = 0; fp = 0; fn = 0
	for i in range(0,len(results)):
		if y_test[i] == results[i]:	# True
			if y_test[i][0] == 1.:		# Positive
				tp += 1
			else:						# Negative
				tn += 1
		else:						# False
			if results[i][0] == 1.:
				fp += 1					# Positive
			else:
				fn += 1					# Negative

	# Calculate confusion matrix
	tpr = tp / (tp + fn)					# True Positive Rate
	tnr = tn / (tn + fp)					# True Negative Rate
	ppv = tp / (tp + fp)					# Sensitivity
	npv = tn / (tn + fn)					# Specificity
	acc = (tp + tn) / (tp + tn + fp + fn)	# Accuracy

	certainty = tf.reduce_mean(tf.abs(tf.subtract(results[:, 0], results[:, 0])))

	print('[ANN] Testing results: ')
	print('[ANN]\tTrue Positive Rate (TPR): {0:.2f}'.format(tpr))
	print('[ANN]\tTrue Negative Rate (TNR): {0:.2f}'.format(tnr))
	print('[ANN]\tSensitivity:              {0:.2f}'.format(ppv))
	print('[ANN]\tSpecificity:              {0:.2f}'.format(npv))
	print('[ANN]\tAccuracy                  {0:.2f}'.format(acc))
	print('[ANN]\tCertainty                 {0:.2f}'.format(certainty))

## Main Program
# Read config file
cfg  = read_config()

# Load mushroom data from dataset
x_train, x_test, y_train, y_test = load_data(cfg['dataset'])
x_train, y_train = shuffle_data(x_train, y_train)
x_test, y_test   = shuffle_data(x_test, y_test)

with tf.Session() as sess:
	if cfg['training']['nn']['train']:
		# Train network on our training data
		print('[ANN] Training network...')
		model = train_network(sess, x_train, y_train, cfg)
	else:
		print('[ANN] Testing network...')
		model = load_model(cfg['training']['nn']['model_dir'])

	# Test network on our testing data
	results = test_network(sess, model, x_test, y_test, cfg)
	analyse_results(y_test, results)
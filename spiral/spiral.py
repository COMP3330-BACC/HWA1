import os 									# os 			-- OS operations, read, write etc
import tensorflow as tf 					# tensorflow 	-- machine learning
import numpy as np 							# numpy			-- python array operations
import matplotlib.pyplot as plt 			# matplotlib 	-- plotting
import csv									# csv 			-- reading from CSVs easily
import yaml									# yaml 			-- reading/writing config files

# Will load data in from the spiral dataset csv as store as x = [(x, y) ...] and y = [(c) ...]
def load_data(data_file):
	with open(data_file, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		x = []
		y = []

		for row in csvreader:
			x.append(list(map(float, row[:-1])))
			y.append([int(row[-1])])
		return x, y
	print('[ERR] Failed to load data from file \'{0}\''.format(data_file))
	exit()

# Reads config parameters from file to store in local config variable
def read_config(cfg_file='config/spiral.yaml'):
	with open(cfg_file, 'r') as yml:
		return yaml.load(yml)
	print('[ERR] Failed to load config file \'{0}\''.format(cfg_file))
	exit()

def plot_data(x, y):
	# Gather points within class a and b for two spiral problem
	# TODO: Find more efficient way to do this
	a = []; b = []
	for i in range(0, len(x)):
		if y[i] == [1]:
			a.append(x[i])
		else:
			b.append(x[i])

	# Plot both classes
	plt.scatter([d[0] for d in a], [d[1] for d in a], color=cfg['two_spiral']['plotting']['c1_colours'][0])
	plt.scatter([d[0] for d in b], [d[1] for d in b], color=cfg['two_spiral']['plotting']['c2_colours'][0])

	# Format plot
	plt.title('Two-Spiral Classification Problem')
	plt.xlabel('x')
	plt.ylabel('y')

def train_network(x, y, cfg):
	## Create network ##
	# Alias config vars
	neurons = cfg['two_spiral']['training']['neurons']
	epochs  = cfg['two_spiral']['training']['epochs']

	# Create placeholders for tensors
	x_ = tf.placeholder(tf.float32, [None, 2], name='x_placeholder')	# Input of (x, y)
	y_ = tf.placeholder(tf.float32, [None, 1], name='y_placeholder')  # Output of [0, 1]

	# First layer
	first_layer_weights = tf.Variable(tf.random_normal([2, neurons]), name='first_weights')
	first_layer_bias    = tf.Variable(tf.random_normal([neurons]), name='first_bias')
	first_layer         = tf.nn.sigmoid(tf.add((tf.matmul(x_, first_layer_weights)), first_layer_bias), name='first')

	# Second layer
	layer_1_weights = tf.Variable(tf.random_normal([neurons, neurons]), name='l1_weights')
	layer_1_bias    = tf.Variable(tf.random_normal([neurons]), name='l1_bias')
	layer_1         = tf.nn.sigmoid(tf.add((tf.matmul(first_layer, layer_1_weights)), layer_1_bias), name='l1')

	# Third layer
	layer_2_weights = tf.Variable(tf.random_normal([neurons, neurons]), name='l2_weights')
	layer_2_bias    = tf.Variable(tf.random_normal([neurons]), name='l2_bias')
	layer_2         = tf.nn.sigmoid(tf.add((tf.matmul(layer_1, layer_2_weights)), layer_2_bias), name='l2')

	# Fourth layer
	layer_3_weights = tf.Variable(tf.random_normal([neurons, neurons]), name='l3_weights')
	layer_3_bias    = tf.Variable(tf.random_normal([neurons]), name='l3_bias')
	layer_3         = tf.nn.sigmoid(tf.add((tf.matmul(layer_2, layer_3_weights)), layer_3_bias), name='l3')

	# Fifth layer
	final_layer_weights = tf.Variable(tf.random_normal([neurons, 1]), name='final_weights')
	final_layer_bias    = tf.Variable(tf.random_normal([1]), name='final_bias')
	final_layer         = tf.nn.sigmoid(tf.add((tf.matmul(layer_3, final_layer_weights)), final_layer_bias), name='final')

	# Define error function
	cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=final_layer))

	# Define optimiser and minimise error function task
	optimiser = tf.train.GradientDescentOptimizer(learning_rate=cfg['two_spiral']['training']['learning_rate']).minimize(cost)

	## Train ##
	# Create error logging storage
	errors = []

	# Create new TF session
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	for i in range(epochs):
		_, error = sess.run([optimiser, cost], feed_dict={x_: x, y_: y})
		errors.append(error)

	# Set size of figure and create first subplot
	fig = plt.figure(figsize=(13, 5))
	plt.subplot(1, 2, 1)

	# Set plot settings
	plt.plot(errors)
	plt.title('Error vs Epoch')
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.grid()

	## Test ##
	# Create second subplot
	plt.subplot(1, 2, 2)

	# Create test data
	lim = cfg['two_spiral']['testing']['limits']
	act_range = np.arange(lim[0], lim[1], 0.1)
	coord = [(x, y) for x in act_range for y in act_range]
	
	# Classify test data
	classifications = np.round(sess.run(final_layer, feed_dict={x_ : coord}))

	# Create class lists and calculate accuracy
	a = []; b = []
	for i in range(0, len(classifications)):
		if classifications[i] == [1]:
			a.append(coord[i])
		else:
			b.append(coord[i])

	# Plot both classes
	plt.scatter([d[0] for d in a], [d[1] for d in a], color=cfg['two_spiral']['plotting']['c1_colours'][1])
	plt.scatter([d[0] for d in b], [d[1] for d in b], color=cfg['two_spiral']['plotting']['c2_colours'][1])

## Main Program

# Read config file
cfg = read_config()

# Load two spiral data from dataset
x, y = load_data(cfg['two_spiral']['dataset'])

# Train network on dataset
train_network(x, y, cfg)

# Create plot of training data and show all plotting
plot_data(x, y)
plt.show()

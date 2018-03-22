import os 						# os 			-- OS operations, read, write etc
import tensorflow as tf 		# tensorflow 	-- machine learning
import numpy as np 				# numpy			-- python array operations
import matplotlib.pyplot as plt # matplotlib 	-- plotting
import csv						# csv 			-- reading from CSVs easily
import yaml						# yaml 			-- reading/writing config files
import time						# time 			-- performance measure
import random
from sklearn.svm import SVC		# sklearn		-- SVM utility

# Reads config parameters from file to store in local config variable
def read_config(cfg_file='config/spiral.yaml'):
	with open(cfg_file, 'r') as yml:
		return yaml.load(yml)
	print('[ERR] Failed to load config file \'{0}\''.format(cfg_file))
	exit()

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

# Plot data for a 2D coordinate vector (x) and class [0, 1] (y)
def plot_data(x, y, train_type, c):
	# Gather points within class a and b for two spiral problem
	# TODO: Find more efficient way to do this
	a = []; b = []
	for i in range(0, len(x)):
		if y[i] == [1]:
			a.append(x[i])
		else:
			b.append(x[i])

	# Plot both classes
	plt.scatter([d[0] for d in a], [d[1] for d in a], color=c[0])
	plt.scatter([d[0] for d in b], [d[1] for d in b], color=c[1])

	# Format plot
	plt.title('Two-Spiral Classification Problem ({0})'.format(train_type))
	plt.xlabel('x')
	plt.ylabel('y')

# ANN
def construct_network(inp, inp_size, out_size, neurons):
	# First layer
	first_layer_weights = tf.Variable(tf.random_normal([inp_size, neurons]), name='first_weights')
	first_layer_bias    = tf.Variable(tf.random_normal([neurons]), name='first_bias')
	first_layer         = tf.nn.sigmoid(tf.add((tf.matmul(inp, first_layer_weights)), first_layer_bias), name='first')

	# Second layer
	l1_weights          = tf.Variable(tf.random_normal([neurons, neurons]), name='l1_weights')
	l1_bias             = tf.Variable(tf.random_normal([neurons]), name='l1_bias')
	l1                  = tf.nn.sigmoid(tf.add((tf.matmul(first_layer, l1_weights)), l1_bias), name='l1')

	# Third layer
	l2_weights          = tf.Variable(tf.random_normal([neurons, neurons]), name='l2_weights')
	l2_bias             = tf.Variable(tf.random_normal([neurons]), name='l2_bias')
	l2                  = tf.nn.sigmoid(tf.add((tf.matmul(l1, l2_weights)), l2_bias), name='l2')

	# Fourth layer
	l3_weights          = tf.Variable(tf.random_normal([neurons, neurons]), name='l3_weights')
	l3_bias             = tf.Variable(tf.random_normal([neurons]), name='l3_bias')
	l3                  = tf.nn.sigmoid(tf.add((tf.matmul(l2, l3_weights)), l3_bias), name='l3')

	# Fifth layer
	final_layer_weights = tf.Variable(tf.random_normal([neurons, out_size]), name='final_weights')
	final_layer_bias    = tf.Variable(tf.random_normal([out_size]), name='final_bias')
	final_layer         = tf.nn.sigmoid(tf.add((tf.matmul(l3, final_layer_weights)), final_layer_bias), name='final')

	return final_layer

def get_fitness(error, time_to_train):
	return error

def train_network(x, y, cfg):
	## Create network ##
	# Alias config vars
	neuron_lims = cfg['two_spiral']['training']['nn']['neurons']
	epoch_lims  = cfg['two_spiral']['training']['nn']['epochs']
	lr_lims     = cfg['two_spiral']['training']['nn']['learning_rate']
	iterations  = cfg['two_spiral']['training']['nn']['iterations']
	acc_thresh  = cfg['two_spiral']['training']['nn']['accuracy_threshold']

	# TODO: Remove when train_network and test_network can be separated
	c1      = cfg['two_spiral']['plotting']['c1']
	c2      = cfg['two_spiral']['plotting']['c2']
	plot_en = cfg['two_spiral']['plotting']['enabled']

	# Create placeholders for tensors
	x_ = tf.placeholder(tf.float32, [None, 2], name='x_placeholder')  # Input of (x, y)
	y_ = tf.placeholder(tf.float32, [None, 1], name='y_placeholder')  # Output of [0, 1]

	opt_model = {'accuracy' : 0., 'fitness' : None}

	# Iterate through models and choose the best one -- evolution!
	# TODO: Use gradient descent with epochs, learning rate and neurons per layer to find better near-model
	while(opt_model['accuracy'] <= acc_thresh):
		# Generate new random learning parameters
		learning_rate = random.uniform(lr_lims[0], lr_lims[1])
		neurons = random.randint(neuron_lims[0], neuron_lims[1])
		epochs = random.randint(epoch_lims[0], epoch_lims[1])

		final_layer = construct_network(x_, 2, 1, neurons)

		# Define error function
		cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=final_layer))

		# Define optimiser and minimise error function task
		optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

		## Train ##
		# Create error logging storage
		errors = []

		# Start timing network creation and training
		t_start = time.time()

		# Create new TF session
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()

		for i in range(epochs):
			_, error = sess.run([optimiser, cost], feed_dict={x_: x, y_: y})
			errors.append(error)
			# Stop model early if we're under an acceptable threshold
			if error <= 1 - acc_thresh:
				print('Stopping model early!')
				epochs = i
				break

		# End time measurement
		t_elapsed = time.time() - t_start

		fitness = get_fitness(error, t_elapsed)

		# Calculate new accuracy #TODO: Calculate error properly
		accuracy = 1 - error

		# If we have a better model, store 
		# TODO store model properly using tf functions
		if(opt_model['fitness'] is None
			or fitness < opt_model['fitness']):
			opt_model = {
				'accuracy'      : accuracy,
				'duration'      : t_elapsed,
				'fitness'		: fitness,
				'epochs'        : epochs,
				'neurons'       : neurons,
				'learning_rate' : learning_rate,
				'errors'        : errors,
				'final_layer'   : final_layer # Should not store model this way -- will be undefined when sess.close is called
			}
			print('[ANN] New model:')
			print('[ANN] \tTraining parameters: epochs={0}, learning_rate={1:.2f}, neurons={2}, fitness={3:.5f}'.format(opt_model['epochs'], opt_model['learning_rate'], opt_model['neurons'], fitness))
			print('[ANN] \tModel accuracy: {0:.3f}%, Time to train: {1:.2f}s'.format(opt_model['accuracy']*100, opt_model['duration']))

	# Set size of figure and create first subplot
	plt.subplot(2, 2, 1)

	# Set plot settings
	plt.plot(opt_model['errors'][:opt_model['epochs']])
	plt.title('Error vs Epoch')
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.grid()

	## Test ##
	# Create second subplot
	plt.subplot(2, 2, 2)

	# Create test data
	lim = cfg['two_spiral']['testing']['limits']
	test_range = np.arange(lim[0], lim[1], 0.1)
	x_test = [(x, y) for x in test_range for y in test_range]
	
	# Start timing the length of time training takes
	t_test = time.time()

	# Classify test data
	y_test = np.round(sess.run(opt_model['final_layer'], feed_dict={x_ : x_test}))

	# Average out the test timing
	t_avg_test = (time.time() - t_test) / float(len(y_test))

	print('[ANN] Average time to test: {0:.2f}us'.format(1000000 * t_avg_test))

	# Create class lists
	plot_data(x_test, y_test, 'ANN', c1)

	# print(opt_model)
	return opt_model

## SVM
def train_svm(x, y, cfg):
	## SVM
	# Read in SVM parameters
	C      = cfg['two_spiral']['training']['svm']['C']
	kernel = cfg['two_spiral']['training']['svm']['kernel']
	gamma  = cfg['two_spiral']['training']['svm']['gamma']

	print('[SVM] Training parameters: C={0}, kernel={1}, gamma={2}'.format(C, kernel, gamma))

	# Create SVM with parameters
	svm = SVC(C=C, kernel=kernel, gamma=gamma)
	
	# Start timing SVM creation and training
	t_start = time.time()
	
	# Train SVM on data
	svm.fit(np.array(x), np.ravel(y))

	# End SVM timing
	t_train = time.time() - t_start

	print('[SVM] Model accuracy: {0:.2f}%, Time to train: {1:.5f}s'.format(100*svm.score(x, np.ravel(y)), t_train))

	return svm

def test_svm(svm, x, cfg):
	# Start timing testing of SVM
	t_start = time.time()

	y = [svm.predict([sample]) for sample in x]

	# Average 
	t_avg_test = (time.time() - t_start) / float(len(y))

	print('[SVM] Average time to test: {0:.2f}us'.format(1000000 * t_avg_test))

	return y

def main():
	# Read config file
	cfg = read_config()

	# Load from config
	lim     = cfg['two_spiral']['testing']['limits']
	c1      = cfg['two_spiral']['plotting']['c1']
	c2      = cfg['two_spiral']['plotting']['c2']
	plot_en = cfg['two_spiral']['plotting']['enabled']

	# Load two spiral data from dataset
	x_train, y_train = load_data(cfg['two_spiral']['dataset'])

	# Create test set
	act_range = np.arange(lim[0], lim[1], 0.1)

	# Perform testing on SVM
	x_test = [(x, y) for x in act_range for y in act_range]

	# Initilise figure size
	if plot_en: 
		fig = plt.figure(figsize=(8, 8))

	## Neural Network
	if cfg['two_spiral']['training']['nn']['enabled']:
		# Train network on dataset
		train_network(x_train, y_train, cfg)

		# Create plot of training data and show all plotting
		if plot_en:
			plot_data(x_train, y_train, 'ANN', c2)
		print()

	## SVM
	if cfg['two_spiral']['training']['svm']['enabled']:
		# Train SVM on training dataset
		svm = train_svm(x_train, y_train, cfg)

		# Test SVM on test dataset
		y_test = test_svm(svm, x_test, cfg)

		# Create plot of training data over testing data
		if plot_en:
			plt.subplot(2, 2, 4)
			plot_data(x_test, y_test, 'SVM', c1)
			plot_data(x_train, y_train, 'SVM', c2)

	plt.show()

main()

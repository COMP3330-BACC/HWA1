import os 						 # os 			-- OS operations, read, write etc
import tensorflow as tf 		 # tensorflow 	-- machine learning
import numpy as np 				 # numpy		-- python array operations
import matplotlib.pyplot as plt  # matplotlib 	-- plotting
import time						 # time 		-- performance measure
import csv
import random
import string

# TODO: Find less hacky way to include parent directories
import sys
sys.path.insert(0, '../../')
import util 					# util 			-- our bag of helper functions!

# Will load data in form as specified by the dataset description 
#	(https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names) 
def load_data(data_file, test_ratio_offset):
	with open(data_file, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		x = []
		y = []

		for row in csvreader:
			for i in range(0, len(row)-1):
				if row[1 + i] == '?':
					row[1 + i] = 'a'

			x_conv = list(map(float, map(ord, row[1:])))
			y_conv = [0. if row[0] == 'p' else 1.]
			# Remove question marks
			for u in x_conv:
				if u == 63.:
					u = -1.
			x.append(x_conv)
			y.append(y_conv)

		split_ratio = 0.9

		x, y = util.shuffle_data(x, y)
		x_train, x_test, y_train, y_test = util.split_data(x, y, split_ratio)
		
		# Check that we can create a confusion matrix 
		# 	(have at least 1 positive and negative sample in test set)

		while(len([result for result in y_test if result[0] == 0.]) < (0.5 - test_ratio_offset) * len(y_test)
			or len([result for result in y_test if result[0] == 1.]) < (0.5 - test_ratio_offset) * len(y_test)):
			x_train, x_test, y_train, y_test = util.split_data(x, y, split_ratio)

		return x_train, x_test, y_train, y_test
	print('[ERR] Failed to load data from file \'{0}\''.format(data_file))
	exit()

def construct_network(inp, weights, biases, neurons):
	fc1 = tf.nn.sigmoid(tf.add((tf.matmul(inp, weights['fc1'])), biases['fc1']), name='fc1')
	fc2 = tf.nn.sigmoid(tf.add((tf.matmul(fc1, weights['fc2'])), biases['fc2']), name='fc2')
	fc3 = tf.nn.sigmoid(tf.add((tf.matmul(fc2, weights['fc3'])), biases['fc3']), name='fc3')
	
	return fc3

def train_network(sess, x, y, cfg):
	# Alias our training config to reduce code
	t_cfg = cfg['nn']

	# Alias config vars to reduce code
	neurons       = t_cfg['parameters']['neurons']
	epochs        = t_cfg['parameters']['epochs']
	learning_rate = t_cfg['parameters']['learning_rate']
	err_thresh    = t_cfg['error_threshold']
	model_dir     = t_cfg['model_dir']
	avg_factor    = t_cfg['avg_factor']
	save_epoch    = t_cfg['save_epoch']
	valid_thresh  = t_cfg['valid_threshold']

	print('[ANN] \tTraining parameters: epochs={0}, learning_rate={1:.2f}, neurons={2}'.format(epochs, learning_rate, neurons))

	# Create validation set
	x_train, x_valid, y_train, y_valid = util.split_data(x, y, 0.9)
	x_valid, y_valid = util.shuffle_data(x_valid, y_valid)

	# Create placeholders for tensors
	x_ = tf.placeholder(tf.float32, [None, 22], name='x_placeholder')
	y_ = tf.placeholder(tf.float32, [None, 1],  name='y_placeholder')

	# Generate new random weights for new network
	weights = {
		'fc1' : tf.Variable(tf.random_normal([22, neurons]),      name='w_fc1'),
		'fc2' : tf.Variable(tf.random_normal([neurons, neurons]), name='w_fc2'),
		'fc3' : tf.Variable(tf.random_normal([neurons, 1]),       name='w_fc3'),
	}

	# Generate new random biases for new network
	biases = {
		'fc1' : tf.Variable(tf.random_normal([neurons]), name='b_fc1'),
		'fc2' : tf.Variable(tf.random_normal([neurons]), name='b_fc2'),
		'fc3' : tf.Variable(tf.random_normal([1]),       name='b_fc3'),
	}

	# Construct our network and return the last layer to output the result
	final_layer = construct_network(x_, weights, biases, neurons)

	# Define error function
	cost_train = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=final_layer))
	cost_valid = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=final_layer))

	# Define optimiser and minimise error function task
	optimiser_train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost_train)
	optimiser_valid = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost_valid)

	# Initialise global variables of the session
	sess.run(tf.global_variables_initializer())

	# Create error logging storage
	train_errors = []
	valid_errors = []

	# Setup our continous plot
	fig = plt.figure()
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

	diff_err = 1.
	vel_err  = 0. 
	acc_err  = 0.

	# Generate a new random model name for new network model
	model_name = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))

	for i in range(epochs):
		# Run network on training and validation sets
		_, train_error = sess.run([optimiser_train, cost_train], feed_dict={x_: x_train, y_: y_train})
		_, valid_error = sess.run([optimiser_train, cost_train], feed_dict={x_: x_valid, y_: y_valid})

		# If we're at a save epoch, save!
		if i % save_epoch == 0:
			model = util.save_model(sess, weights, biases, neurons, train_errors, os.path.join(model_dir, model_name + "_model"))

		# Add new errors to list
		train_errors.append(train_error)
		valid_errors.append(valid_error)

		# If we have at least an averageable amount of samples
		if i > avg_factor:
			avg_train_error = 0; avg_valid_error = 0
			# Get sum over last n epochs
			for j in range(0, avg_factor):
				avg_train_error += train_errors[i - j]
				avg_valid_error += valid_errors[i - j]
			# Average them
			avg_train_error /= avg_factor
			avg_valid_error /= avg_factor

			# Calculate change in velocity of error difference
			acc_err = vel_err - (diff_err - abs(avg_valid_error - avg_train_error))

			# Calculate change in error difference (positive -> convergence, negative -> divergence)
			vel_err = diff_err - abs(avg_valid_error - avg_train_error)
			
			# Calculate error difference between validation and training
			diff_err = abs(avg_valid_error - avg_train_error)
			# print('[ANN] Epoch: {0:4d}, Δerr = {1:7.4f}, 𝛿(Δerr) = {2:7.4f}, 𝛿(𝛿(Δerr)) = {3:7.4f}'.format(i, diff_err, vel_err, acc_err)) # DEBUG

		# If we already have our target error, terminate early
		if train_error <= err_thresh or (diff_err > valid_thresh and vel_err < 0.):
			break
		
		# Set plot settings
		if i > 0:
			plt.plot(train_errors[:epochs], color='r', label='training')
			plt.plot(valid_errors[:epochs], color='b', label='validation')
			plt.axis([0, i, 0., 1.])
			plt.draw()
			plt.pause(0.001)			

	plt.ioff()

	t_elapsed = time.time() - t_start

	# Calculate new simple accuracy from final error
	accuracy = 1 - train_error

	# Save model to file
	model = util.save_model(sess, weights, biases, neurons, train_errors, os.path.join(model_dir, model_name + "_model"))

	print('\n[ANN] Training Completed:')

	# Calculate number of minutes, seconds and milliseconds elapsed
	t_m  = t_elapsed / 60
	t_s  = t_elapsed % 60
	t_ms = (t_s % 1) * 1000

	print('[ANN]\tModel name: {0}'.format(model_name))
	print('[ANN]\tSimple model accuracy: {0:.3f}%'.format(accuracy*100))
	print('[ANN]\tTime elapsed: {0:2d}m {1:2d}s {2:3d}ms'.format(int(t_m), int(t_s), int(t_ms)))

	return model, model_name, {'num_layers': len(weights), 'layer_width': neurons, 'learning_rate': learning_rate, 'time_to_train': t_elapsed, 'train_errors': [float(i) for i in train_errors], 'valid_errors': [float(i) for i in valid_errors]}

# Test network with our test set, return the test results from the test input
def test_network(sess, model, x_test, y_test, cfg):
	x_t = tf.placeholder(tf.float32, [None, 22], name='t_x_placeholder')
	y_t = tf.placeholder(tf.float32, [None, 1],  name='t_y_placeholder')

	final_layer = construct_network(x_t, model['weights'], model['biases'], model['neurons'])

	# Start timing the length of time training takes
	t_test = time.time()

	# Classify test data
	result = sess.run(final_layer, feed_dict={x_t : x_test, y_t : y_test})

	# Average out the test timing
	t_avg_test = (time.time() - t_test) / float(len(y_test))

	print('[ANN] Testing Completed:')
	print('[ANN]\tAverage time to test: {0:.2f}us'.format(1000000 * t_avg_test))
	return result

def plot_roc(sk_fpr, sk_tpr, roc_auc):
	fig = plt.figure()
	plt.title('ROC Curve')
	plt.plot([0., 1.], [0., 1.], 'r--', label='ROC Curve (area = {0:.2f})'.format(roc_auc))
	plt.plot(sk_fpr, sk_tpr, 'b')

	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.axis([0, 1., 0., 1.])
	plt.legend()
	plt.show()

## Main Program
def main():
	# Read config file
	cfg  = util.read_config('config/mushroom.yaml')

	# Load mushroom data from dataset
	x_train, x_test, y_train, y_test = load_data(cfg['dataset'], cfg['test_ratio_offset'])
	x_train, y_train = util.shuffle_data(x_train, y_train)
	x_test, y_test   = util.shuffle_data(x_test, y_test)

	# Default model name as loaded from file, overwritten if training
	model_name = cfg['nn']['model_name']
	model_dir  = cfg['nn']['model_dir']

	with tf.Session() as sess:
		if cfg['nn']['train']:
			# Train network on our training data
			print('[ANN] Training new network...')
			model, model_name, train_stats = train_network(sess, x_train, y_train, cfg)
		else:
			loaded_results = util.load_results(os.path.join(model_dir, model_name + "_cm"))
			# Setup our continous plot
			plt.title('Error vs Epoch')
			plt.plot(loaded_results['train_stats']['train_errors'], color='r', label='training')
			plt.plot(loaded_results['train_stats']['valid_errors'], color='b', label='validation')
			plt.xlabel('Epoch')
			plt.ylabel('Error')
			plt.legend()
			plt.grid()
			plt.show()

			print('[ANN] Testing network {0}...'.format(model_name))
			model = util.load_model(os.path.join(model_dir, model_name + "_model"))
			train_stats = loaded_results['train_stats']

		# Test network on our testing data
		results = test_network(sess, model, x_test, y_test, cfg)
		conf_mat, sk_fpr, sk_tpr, roc_auc = util.analyse_results(y_test, results)
		print('[ANN] ROC Area Under Curve: {0:.2f}'.format(roc_auc))
		plot_roc(sk_fpr, sk_tpr, roc_auc)
		results_to_save = {'conf_mat': conf_mat, 'train_stats': train_stats, 'roc_auc' : float(roc_auc)}
		util.store_results(results_to_save, os.path.join(model_dir, model_name + "_cm"))

# Make sure to only run if not being imported
if __name__ == '__main__':
	main()
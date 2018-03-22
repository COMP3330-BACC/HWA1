import os 						 # os 			-- OS operations, read, write etc
import tensorflow as tf 		 # tensorflow 	-- machine learning
import numpy as np 				 # numpy		-- python array operations
import matplotlib.pyplot as plt  # matplotlib 	-- plotting
import yaml						 # yaml 		-- reading/writing config files
import time						 # time 		-- performance measure
import csv
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

def construct_network(inp_placeholder, inp_size, out_size, neurons):
	# First layer
	first_layer_weights = tf.Variable(tf.random_normal([inp_size, neurons]), name='first_weights')
	first_layer_bias    = tf.Variable(tf.random_normal([neurons]),     name='first_bias')
	first_layer         = tf.nn.sigmoid(tf.add((tf.matmul(inp_placeholder, first_layer_weights)), first_layer_bias), name='first')

	# Second layer
	l1_weights          = tf.Variable(tf.random_normal([neurons, neurons]), name='l1_weights')
	l1_bias             = tf.Variable(tf.random_normal([neurons]),     name='l1_bias')
	l1                  = tf.nn.sigmoid(tf.add((tf.matmul(first_layer, l1_weights)), l1_bias), name='l1')

	# Third layer
	l2_weights          = tf.Variable(tf.random_normal([neurons, neurons]), name='l2_weights')
	l2_bias             = tf.Variable(tf.random_normal([neurons]),     name='l2_bias')
	l2                  = tf.nn.sigmoid(tf.add((tf.matmul(l1, l2_weights)), l2_bias), name='l2')

	# Fourth layer
	l3_weights          = tf.Variable(tf.random_normal([neurons, neurons]), name='l3_weights')
	l3_bias             = tf.Variable(tf.random_normal([neurons]),     name='l3_bias')
	l3                  = tf.nn.sigmoid(tf.add((tf.matmul(l2, l3_weights)), l3_bias), name='l3')

	# Fifth layer
	l4_weights          = tf.Variable(tf.random_normal([neurons, neurons]), name='l4_weights')
	l4_bias             = tf.Variable(tf.random_normal([neurons]),     name='l4_bias')
	l4                  = tf.nn.sigmoid(tf.add((tf.matmul(l3, l4_weights)), l4_bias), name='l4')

	# Fifth layer
	l5_weights          = tf.Variable(tf.random_normal([neurons, neurons]), name='l5_weights')
	l5_bias             = tf.Variable(tf.random_normal([neurons]),     name='l5_bias')
	l5                  = tf.nn.sigmoid(tf.add((tf.matmul(l4, l5_weights)), l5_bias), name='l5')

	# Fifth layer
	l6_weights          = tf.Variable(tf.random_normal([neurons, neurons]), name='l6_weights')
	l6_bias             = tf.Variable(tf.random_normal([neurons]),     name='l6_bias')
	l6                  = tf.nn.sigmoid(tf.add((tf.matmul(l5, l6_weights)), l6_bias), name='l6')

	# Sixth layer
	final_layer_weights = tf.Variable(tf.random_normal([neurons, out_size]),  name='final_weights')
	final_layer_bias    = tf.Variable(tf.random_normal([out_size]),      name='final_bias')
	final_layer         = tf.nn.sigmoid(tf.add((tf.matmul(l6, final_layer_weights)), final_layer_bias), name='final')

	return final_layer

def train_network(x, y, cfg):
	# Alias config vars
	neuron_lims = cfg['mushroom']['training']['nn']['neurons']
	epoch_lims  = cfg['mushroom']['training']['nn']['epochs']
	lr_lims     = cfg['mushroom']['training']['nn']['learning_rate']
	iterations  = cfg['mushroom']['training']['nn']['iterations']
	acc_thresh  = cfg['mushroom']['training']['nn']['accuracy_threshold']

	# Create placeholders for tensors
	x_ = tf.placeholder(tf.float32, [None, 22], name='x_placeholder')
	y_ = tf.placeholder(tf.float32, [None, 1],  name='y_placeholder')

	opt_model = {'accuracy' : 0}

	while(opt_model['accuracy'] <= acc_thresh):
		# Generate new random learning parameters
		learning_rate = random.uniform(lr_lims[0], lr_lims[1])
		neurons = random.randint(neuron_lims[0], neuron_lims[1])
		epochs = random.randint(epoch_lims[0], epoch_lims[1])

		final_layer = construct_network(x_, 22, 1, neurons)
		
		# Define error function
		cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=final_layer))

		# Define optimiser and minimise error function task
		optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

		## Train ##
		# Create error logging storage
		errors = []

		t_start = time.time()

		# Create new TF session
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()

		for i in range(epochs):
			_, error = sess.run([optimiser, cost], feed_dict={x_: x, y_: y})
			errors.append(error)

		t_elapsed = time.time() - t_start

		# Calculate new accuracy
		accuracy = 1 - error

		# If we have a better model, store 
		if(accuracy > opt_model['accuracy'] 
			or accuracy >= opt_model['accuracy'] and t_elapsed < opt_model['duration']):
			opt_model = {
				'accuracy'      : accuracy,
				'duration'      : t_elapsed,
				'epochs'        : epochs,
				'neurons'       : neurons,
				'learning_rate' : learning_rate,
				'errors'        : errors,
				'final_layer'   : final_layer
			}
			print('[ANN] New model:')
			print('[ANN] \tTraining parameters: epochs={0}, learning_rate={1:.2f}, neurons={2}'.format(opt_model['epochs'], opt_model['learning_rate'], opt_model['neurons']))
			print('[ANN] \tModel accuracy: {0:.3f}%, Time elapsed: {1:.2f}s'.format(opt_model['accuracy']*100, opt_model['duration']))

	# Set plot settings
	plt.plot(opt_model['errors'][:epochs])
	plt.title('Error vs Epoch')
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.grid()


def test_network(x, y, cfg):
	print('tested')

## Main Program

# Read config file
cfg  = read_config()

# Load mushroom data from dataset
x_train, x_test, y_train, y_test = load_data(cfg['mushroom']['dataset'])

# Train network on our training data
model = train_network(x_train, y_train, cfg)

# Test network on our testing data
test_network(x_test, y_test, cfg)
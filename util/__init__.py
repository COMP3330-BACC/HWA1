import pickle
import random
import yaml

### Network Utilities
# Save model weights, biases and neurons per layer to file at <path>
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

# Load model weights, biases and neurons per layer to file from <path>
def load_model(path):
	with open(path, "rb") as pf:
		model = pickle.load(pf)
		return model
	print('ERR: No model found at \'{0}\', have you trained a model yet?'.format(path))

# Analyse the results of testing and return our confusion matrix
def analyse_results(y_test, results):
	# Make variables for true pos, true neg, false pos, false neg
	tp = 0; tn = 0; fp = 0; fn = 0
	for i in range(0, len(results)):
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

	if tp + fn == 0 or tn + fp == 0 or tp + fp == 0 or tn + fn == 0:
		print('ERR: Cannot calculate confusion matrix')
		return None
	else:
		# Calculate confusion matrix
		tpr = tp / (tp + fn)					# True Positive Rate
		tnr = tn / (tn + fp)					# True Negative Rate
		ppv = tp / (tp + fp)					# Sensitivity
		npv = tn / (tn + fn)					# Specificity
		acc = (tp + tn) / (tp + tn + fp + fn)	# Accuracy

		print('[ANN] Testing results: ')
		print('[ANN]\tTrue Positive Rate (TPR): {0:.2f}'.format(tpr))
		print('[ANN]\tTrue Negative Rate (TNR): {0:.2f}'.format(tnr))
		print('[ANN]\tSensitivity:              {0:.2f}'.format(ppv))
		print('[ANN]\tSpecificity:              {0:.2f}'.format(npv))
		print('[ANN]\tAccuracy                  {0:.2f}'.format(acc))
		return {
			'tpr' : tpr,
			'tnr' : tnr,
			'ppv' : ppv,
			'npv' : npv,
			'acc' : acc
		}

# Store our results analysis
def store_results(conf_mat, conf_file):
	with open(conf_file, 'w') as yml:
		yaml.dump(conf_mat, yml)
		return
	print('[ERR] Failed to load config file \'{0}\''.format(conf_file))
	exit()

### Dataset Utilities

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

### Configuration Utilities
# Read config file and return yaml dictionary of parameters
def read_config(cfg_file):
	with open(cfg_file, 'r') as yml:
		return yaml.load(yml)
	print('[ERR] Failed to load config file \'{0}\''.format(cfg_file))
	exit()
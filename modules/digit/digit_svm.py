import os 						 # os 			-- OS operations, read, write etc
import tensorflow as tf 		 # tensorflow 	-- machine learning
import numpy as np 				 # numpy		-- python array operations
import matplotlib.pyplot as plt  # matplotlib 	-- plotting
import time						 # time 		-- performance measure
import csv
import random
import string
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder


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

def train_svm_multi (x, y, cfg):
	C      = cfg['svm']['C']
	kernel = cfg['svm']['kernel']
	gamma  = cfg['svm']['gamma']
	n_estimators = 20;

	print('\t[SVM] Training parameters: C={0}, kernel={1}, gamma={2}'.format(C, kernel, gamma))
	svm = BaggingClassifier(SVC(C=C, kernel=kernel, gamma=gamma), max_samples=1.0/n_estimators, n_estimators=n_estimators)
	t_start=time.time()
	svm.fit(np.array(x), np.ravel(y))
	t_train = time.time()-t_start
	print('\t[SVM] Model accuracy: {0:.2f}%, Time to train: {1:.5f}s'.format(100*svm.score(x, np.ravel(y)), t_train))
	
	return svm

def test_svm(svm, x):
	t_start = time.time()
	y = svm.predict(x)
	t_avg_test = (time.time() - t_start)/float(len(y))

	print('\t[SVM] Average time to test: {0:.2f}us'.format(1000000 * t_avg_test))
	return y

def returnNotMatches(a, b):
    return [[x for x in a if x not in b], [x for x in b if x not in a]]

def confusion(y1, y2):
	classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

	confusion_matrix = np.zeros((len(classes), len(classes)))

	for true_label, predicted_label in zip(y1, y2):
		confusion_matrix[true_label][predicted_label] += 1
	return confusion_matrix

def plot_confusion_matrix(y1, y2, title='Confusion matrix', cmap=plt.cm.gray_r):
   
	y_actual = pd.Series(y1, name='Actual')
	y_predicted = pd.Series(y2, name='Predicted')
	df_confusion = pd.crosstab(y_actual, y_predicted, rownames=['Actual'], colnames=['Predicted'], margins=True)

	plt.matshow(df_confusion, cmap=cmap) # imshow
	#plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(df_confusion.columns))
	plt.xticks(tick_marks, df_confusion.columns, rotation=45)
	plt.yticks(tick_marks, df_confusion.index)
	#plt.tight_layout()
	plt.ylabel(df_confusion.index.name)
	plt.xlabel(df_confusion.columns.name)
	plt.show()

def plot_roc(y1, y2, n_classes):
	fpr = dict()
	tpr = dict()
	roc_auc = dict()

	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(y1, y2)
		roc_auc[i] = auc(fpr[i], tpr[i])

	fpr["micro"], tpr["micro"], _ = roc_curve(y1.ravel(), y2.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
		mean_tpr += interp(all_fpr, fpr[i], tpr[i])
	mean_tpr /= n_classes
	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

	plt.figure()
	lw=2

	colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
	for i, color in zip(range(n_classes), colors):
	    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
	             label='ROC curve of class {0} (area = {1:0.2f})'
	             ''.format(i, roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Multi-class receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.show()


def main():
	cfg = util.read_config('config/digit.yaml')

	x_train, x_test, y_train, y_test = load_data(cfg['dataset'])
	x_train, y_train = util.shuffle_data(x_train, y_train)
	x_test, y_test   = util.shuffle_data(x_test, y_test)

	model_name = cfg['svm']['model_name']
	model_dir = cfg['svm']['model_dir']

	svm = train_svm_multi(x_train, y_train, cfg)
	y_test2 = test_svm(svm, x_test)
	
	i=0
	y_test3 = []
	while i<len(y_test):
		y_test3.append(y_test[i][0])
		i += 1
	y_test3 = np.array(y_test3)
	
	class_labels = LabelEncoder()
	y_actual = class_labels.fit_transform(y_test3)
	y_predictions = class_labels.fit_transform(y_test2)

	plot_confusion_matrix(y_actual, y_predictions)
	plot_roc(y_actual, y_predictions, 10)

if __name__ == '__main__':
	main()	
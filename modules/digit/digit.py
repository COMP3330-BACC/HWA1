import os  # os 			-- OS operations, read, write etc
import tensorflow as tf  # tensorflow 	-- machine learning
import numpy as np  # numpy		-- python array operations
import matplotlib.pyplot as plt  # matplotlib 	-- plotting
import time  # time 		-- performance measure
import csv
import random
import string
import pandas as pd
from sklearn.svm import SVC
from sklearn import preprocessing

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

        y = relabel(y)

        x, y = util.shuffle_data(x, y)
        x_train, x_test, y_train, y_test = util.split_data(x, y, 0.9)

        return x_train, x_test, y_train, y_test
    print('[ERR] Failed to load data from file \'{0}\''.format(data_file))
    exit()


def relabel(data):
    new_data = []
    classes = 10
    for row in data:
        new_row = [0] * classes
        new_row[int(row[0])] = 1
        new_data.append(new_row)
    return new_data


def construct_network(inp, weights, biases, neurons):
    fc1 = tf.nn.sigmoid(
        tf.add((tf.matmul(inp, weights['fc1'])), biases['fc1']), name='fc1')
    fc2 = tf.nn.sigmoid(
        tf.add((tf.matmul(fc1, weights['fc2'])), biases['fc2']), name='fc2')
    fc3 = tf.nn.sigmoid(
        tf.add((tf.matmul(fc2, weights['fc3'])), biases['fc3']), name='fc3')
    fc4 = tf.nn.sigmoid(
        tf.add((tf.matmul(fc3, weights['fc4'])), biases['fc4']), name='fc4')
    fc5 = tf.nn.sigmoid(
        tf.add((tf.matmul(fc4, weights['fc5'])), biases['fc5']), name='fc5')
    fc6 = tf.nn.sigmoid(
        tf.add((tf.matmul(fc4, weights['fc6'])), biases['fc6']), name='fc5')
    fc7 = tf.nn.sigmoid(
        tf.add((tf.matmul(fc5, weights['fc7'])), biases['fc7']), name='fc6')

    return fc7


def train_network(sess, x, y, cfg):
    # Alias our training config to reduce code
    t_cfg = cfg['nn']

    # Alias config vars to reduce code
    neurons = t_cfg['parameters']['neurons']
    epochs = t_cfg['parameters']['epochs']
    learning_rate = t_cfg['parameters']['learning_rate']
    err_thresh = t_cfg['error_threshold']
    model_dir = t_cfg['model_dir']
    avg_factor = t_cfg['avg_factor']
    save_epoch = t_cfg['save_epoch']

    print(
        '[ANN] \tTraining parameters: epochs={0}, learning_rate={1:.3f}, neurons={2}'.
        format(epochs, learning_rate, neurons))

    # Create validation set
    x_train, x_valid, y_train, y_valid = util.split_data(x, y, 0.95)
    x_valid, y_valid = util.shuffle_data(x_valid, y_valid)

    # Create placeholders for tensors
    x_ = tf.placeholder(tf.float32, [None, 784], name='x_placeholder')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_placeholder')

    # Generate new random weights for new network
    weights = {
        'fc1': tf.Variable(tf.random_normal([784, neurons]), name='w_fc1'),
        'fc2': tf.Variable(tf.random_normal([neurons, neurons]), name='w_fc2'),
        'fc3': tf.Variable(tf.random_normal([neurons, neurons]), name='w_fc3'),
        'fc4': tf.Variable(tf.random_normal([neurons, neurons]), name='w_fc4'),
        'fc5': tf.Variable(tf.random_normal([neurons, neurons]), name='w_fc5'),
        'fc6': tf.Variable(tf.random_normal([neurons, neurons]), name='w_fc6'),
        'fc7': tf.Variable(tf.random_normal([neurons, 10]), name='w_fc7'),
    }

    # Generate new random biases for new network
    biases = {
        'fc1': tf.Variable(tf.random_normal([neurons]), name='b_fc1'),
        'fc2': tf.Variable(tf.random_normal([neurons]), name='b_fc2'),
        'fc3': tf.Variable(tf.random_normal([neurons]), name='b_fc3'),
        'fc4': tf.Variable(tf.random_normal([neurons]), name='b_fc4'),
        'fc5': tf.Variable(tf.random_normal([neurons]), name='b_fc5'),
        'fc6': tf.Variable(tf.random_normal([neurons]), name='b_fc6'),
        'fc7': tf.Variable(tf.random_normal([10]), name='b_fc7'),
    }

    # Construct our network and return the last layer to output the result
    final_layer = construct_network(x_, weights, biases, neurons)

    # Define error function
    cost = tf.reduce_mean(
        tf.losses.mean_squared_error(labels=y_, predictions=final_layer))

    # Define optimiser and minimise error function task
    optimiser = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(cost)

    # Initialise global variables of the session
    sess.run(tf.global_variables_initializer())

    # Create error logging storage
    train_errors = []
    valid_errors = []

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

    diff_err = 1.
    vel_err = 0.
    acc_err = 0.

    # Generate a new random model name for new network model
    model_name = ''.join(
        random.choice(string.ascii_lowercase + string.digits)
        for _ in range(4))

    for i in range(epochs):
        # Run network on training and validation sets
        _, train_error = sess.run(
            [optimiser, cost], feed_dict={
                x_: x_train,
                y_: y_train
            })
        _, valid_error = sess.run(
            [optimiser, cost], feed_dict={
                x_: x_valid,
                y_: y_valid
            })

        # If we're at a save epoch, save!
        if i % save_epoch == 0:
            model = util.save_model(sess, weights, biases, neurons, train_errors,
                                    os.path.join(model_dir,
                                                 model_name + "_model"))

        # Add new errors to list
        train_errors.append(train_error)
        valid_errors.append(valid_error)

        # If we have at least an averageable amount of samples
        if i > avg_factor:
            avg_train_error = 0
            avg_valid_error = 0
            # Get sum over last n epochs
            for j in range(0, avg_factor):
                avg_train_error += train_errors[i - j]
                avg_valid_error += valid_errors[i - j]
            # Average them
            avg_train_error /= avg_factor
            avg_valid_error /= avg_factor

            # Calculate change in velocity of error difference
            acc_err = vel_err - (
                diff_err - abs(avg_valid_error - avg_train_error))

            # Calculate change in error difference (positive -> convergence, negative -> divergence)
            vel_err = diff_err - abs(avg_valid_error - avg_train_error)

            # Calculate error difference between validation and training
            diff_err = abs(avg_valid_error - avg_train_error)
            # print('[ANN] Epoch: {0:4d}, Δerr = {1:7.4f}, 𝛿(Δerr) = {2:7.4f}, 𝛿(𝛿(Δerr)) = {3:7.4f}'.format(i, diff_err, vel_err, acc_err)) # DEBUG

        # If we already have our target error, terminate early
        if train_error <= err_thresh:
            break

        # Set plot settings
        if i > 0:
            plt.plot(train_errors[:epochs], color='r', label='training')
            plt.plot(valid_errors[:epochs], color='b', label='validation')
            plt.axis([0, i, 0., 1.])
            plt.draw()
            plt.pause(0.001)

    t_elapsed = time.time() - t_start

    # Calculate new simple accuracy from final error
    accuracy = 1 - train_error

    # Save model to file
    model = util.save_model(sess, weights, biases, neurons, train_errors,
                            os.path.join(model_dir, model_name + "_model"))

    print('\n[ANN] Training Completed:')

    # Calculate number of minutes, seconds and milliseconds elapsed
    t_m = t_elapsed / 60
    t_s = t_elapsed % 60
    t_ms = (t_s % 1) * 1000

    print('[ANN]\tModel name: {0}'.format(model_name))
    print('[ANN]\tSimple model accuracy: {0:.3f}%'.format(accuracy * 100))
    print('[ANN]\tTime elapsed: {0:2d}m {1:2d}s {2:3d}ms'.format(
        int(t_m), int(t_s), int(t_ms)))

    return model, model_name


# Test network with our test set, return the test results from the test input
def test_network(sess, model, x_test, y_test, cfg):
    x_t = tf.placeholder(tf.float32, [None, 784], name='t_x_placeholder')
    y_t = tf.placeholder(tf.float32, [None, 10], name='t_y_placeholder')

    final_layer = construct_network(x_t, model['weights'], model['biases'],
                                    model['neurons'])

    # Start timing the length of time training takes
    t_test = time.time()

    # Classify test data
    result = sess.run(final_layer, feed_dict={x_t : x_test, y_t : y_test})

    # Average out the test timing
    t_avg_test = (time.time() - t_test) / float(len(y_test))

    print('[ANN] Testing Completed:')
    print('[ANN]\tAverage time to test: {0:.2f}us'.format(
        1000000 * t_avg_test))
    return result

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

def convert_to_nonbinary(y_test2):
    y_test3 = []
    i=0
    while i<len(y_test2):
        j=0
        cont = True
        while (j<len(y_test2[i])) and (cont == True):
            if y_test2[i][j] == 1:
                y_test3.append(j)
                cont = False
            j += 1
        i += 1
    return y_test3

def convert_to_different_nonbinary(y_in):
    y_out = []
    i=0
    while i<len(y_in):
        j=0
        biggest1 = 0
        biggest2 = 0
        while (j<10):
            if y_in[i][j] > biggest1:
                biggest1 = y_in[i][j]
                biggest2 = j
            j += 1
        y_out.append(biggest2)
        i += 1
    return y_out


## Main Program
def main():
    # Read config file
    cfg = util.read_config('config/digit.yaml')

    # Load digit data from dataset
    x_train, x_test, y_train, y_test = load_data(cfg['dataset'])
    x_train, y_train = util.shuffle_data(x_train, y_train)
    x_test, y_test = util.shuffle_data(x_test, y_test)

    # Default model name as loaded from file, overwritten if training
    model_name = cfg['nn']['model_name']
    model_dir = cfg['nn']['model_dir']

    with tf.Session() as sess:
        if cfg['nn']['train']:
            # Train network on our training data
            print('[ANN] Training new network...')
            model, model_name = train_network(sess, x_train, y_train, cfg)
        else:
            print('[ANN] Testing network {0}...'.format(model_name))
            model = util.load_model(
                os.path.join(model_dir, model_name + "_model"))

        # Test network on our testing data
        results = test_network(sess, model, x_test, y_test, cfg)

        # TODO: Tristan to reimplement analyse results to get confusion matrix and roc curve 
        conf_mat={}
        # conf_mat = util.analyse_results(y_test, results)
        util.store_results(conf_mat, os.path.join(model_dir,
                                                  model_name + "_cm"))
        
        y_test2 = np.array(y_test)       
        
        y_test3 = convert_to_nonbinary(y_test2)
    #    print(y_test3)

        print(results)
        y_results = convert_to_different_nonbinary(results)
        print(y_results)

    #    plot_confusion_matrix(y_test3, results)

'''	'''
# Make sure to only run if not being imported
if __name__ == '__main__':
    main()
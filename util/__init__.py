import pickle
import random
import yaml
import string


### Network Utilities
# Save model weights, biases and neurons per layer to file at <path>
def save_model(sess, weights, biases, neurons, path):
    with open(path + ".yaml", "wb") as pf:
        # Create new dictionary containers for non-tf types
        n_weights = {}
        n_biases = {}
        for key in weights.keys():
            n_weights.update({key: sess.run(weights[key])})
        for key in biases.keys():
            n_biases.update({key: sess.run(biases[key])})
        model = {'weights': n_weights, 'biases': n_biases, 'neurons': neurons}
        pickle.dump(model, pf)
        return model


# Load model weights, biases and neurons per layer to file from <path>
def load_model(path):
    with open(path + ".yaml", "rb") as pf:
        model = pickle.load(pf)
        return model
    print(
        'ERR: No model found at \'{0}\', have you trained a model yet?'.format(
            path))


# Analyse the results of testing and return our confusion matrix
def analyse_results(y_test, results):
    # Make variables for true pos, true neg, false pos, false neg
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, len(results)):
        [exp_result, act_result] = [None, None]
        if type(y_test[i]) is list:
            exp_result = y_test[i][0]
            act_result = results[i][0]
        else:
            exp_result = y_test[i]
            act_result = results[i]

        if exp_result == act_result:  # True
            if exp_result == 1.:  # Positive
                tp += 1
            else:  # Negative
                tn += 1
        else:
            if act_result == 1.:
                fp += 1  # Positive
            else:
                fn += 1  # Negative

    ## Calculate confusion matrix
    # Store our errors in a list
    err_list = []

    # True Positive Rate
    if tp + fn == 0:
        err_list.append('Sum of true positives and false negatives is zero')
    else:
        tpr = tp / (tp + fn)

    # True Negative Rate
    if tn + fp == 0:
        err_list.append('Sum of true negatives and false positives is zero')
    else:
        tnr = tn / (tn + fp)

    # Sensitivity
    if tp + fp == 0:
        err_list.append('Sum of true positives and false positives is zero')
    else:
        ppv = tp / (tp + fp)

    # Specificity
    if tn + fn == 0:
        err_list.append('Sum of true negatives and false negatives is zero')
    else:
        npv = tn / (tn + fn)

    # Accuracy
    acc = (tp + tn) / (tp + tn + fp + fn)

    if len(err_list) > 0:
        print('ERR: Cannot calculate complete confusion matrix:')
        for e in err_list:
            print('\t-{0}'.format(e))
    else:

        print('[ANN] Testing results: ')
        print('[ANN]\tTrue Positive Rate (TPR): {0:.2f}'.format(tpr))
        print('[ANN]\tTrue Negative Rate (TNR): {0:.2f}'.format(tnr))
        print('[ANN]\tSensitivity:              {0:.2f}'.format(ppv))
        print('[ANN]\tSpecificity:              {0:.2f}'.format(npv))
        print('[ANN]\tAccuracy                  {0:.2f}'.format(acc))
        return {'tpr': tpr, 'tnr': tnr, 'ppv': ppv, 'npv': npv, 'acc': acc}


# Store our results analysis
def store_results(conf_mat, conf_file):
    with open(conf_file + ".yaml", 'w') as yml:
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
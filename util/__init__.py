import pickle
import random
import yaml
import string
import matplotlib.pyplot as plt  # matplotlib   -- plotting
from sklearn.metrics import roc_curve, auc


### Network Utilities
# Save model weights, biases and neurons per layer to file at <path>
def save_model(sess, weights, biases, neurons, errors, path):
    with open(path + ".yaml", "wb") as pf:
        # Create new dictionary containers for non-tf types
        n_weights = {}
        n_biases = {}
        for key in weights.keys():
            n_weights.update({key: sess.run(weights[key])})
        for key in biases.keys():
            n_biases.update({key: sess.run(biases[key])})
        model = {'weights': n_weights, 'biases': n_biases, 'num_layers': len(weights), 'neurons': neurons, 'errors': errors}
        pickle.dump(model, pf)
        return model


# Load model weights, biases and neurons per layer to file from <path>
def load_model(path):
    try:
        with open(path + ".yaml", "rb") as pf:
            model = pickle.load(pf)
            return model
    except FileNotFoundError:
        print('[ERR] No model found at \'{0}\'. (Have you trained the model yet?)'.format(path))
        exit()


# Analyse the results of testing and return our confusion matrix
def analyse_results(y_test, results):
    # Make variables for true pos, true neg, false pos, false neg
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # Evaluate the ROC curve 
    sk_fpr, sk_tpr, _ = roc_curve(y_test, results)
    roc_auc = auc(sk_fpr, sk_tpr)

    for i in range(0, len(results)):
        [exp_result, act_result] = [None, None]
        if type(y_test[i]) is list:
            exp_result = y_test[i][0]
            act_result = round(results[i][0])
        else:
            exp_result = y_test[i]
            act_result = round(results[i])

        if exp_result == act_result:  # True
            if exp_result == 1.:  # Positive
                tp += 1
            else:  # Negative
                tn += 1
        elif exp_result != act_result:
            if exp_result == 1.:
                fn += 1  # Negative
            else:
                fp += 1  # Positive

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
        print('[ERR] Cannot calculate complete confusion matrix:')
        for e in err_list:
            print('\t-{0}'.format(e))
        return {}, sk_fpr, sk_tpr, roc_auc
    else:
        print('[ANN] Testing results: ')
        print('[ANN]\tTrue Positive Rate (TPR): {0:.2f}'.format(tpr))
        print('[ANN]\tTrue Negative Rate (TNR): {0:.2f}'.format(tnr))
        print('[ANN]\tSensitivity:              {0:.2f}'.format(ppv))
        print('[ANN]\tSpecificity:              {0:.2f}'.format(npv))
        print('[ANN]\tAccuracy                  {0:.2f}'.format(acc))
        return {'tpr': tpr, 'tnr': tnr, 'ppv': ppv, 'npv': npv, 'acc': acc}, sk_fpr, sk_tpr, roc_auc


# Store our results analysis
def store_results(results, conf_file):
    try:
        with open(conf_file + ".yaml", 'w') as yml:
            yaml.dump(results, yml, default_flow_style=False)
            return
    except FileNotFoundError:
        print('[ERR] Failed to load config file \'{0}\''.format(conf_file))
        exit()

# Load our results analysis
def load_results(conf_file):
    try:
        with open(conf_file + ".yaml", 'r') as yml:
            return yaml.load(yml)
    except FileNotFoundError:
        print('[ERR] Failed to load results file \'{0}\''.format(conf_file))
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
    try:
        with open(cfg_file, 'r') as yml:
            return yaml.load(yml)
    except FileNotFoundError:
        print('[ERR] Failed to load config file \'{0}\''.format(cfg_file))
        exit()
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

import tqdm
import math
import time
from collections import Counter, deque, OrderedDict

from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Network
from maml import MAML

import cvxpy as cp


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Enable cuda if available

def group_by_label(y):
    """ Groups data by label and returns indices per label """
    label_dict = {}
    for i in range(len(y)):
        if y[i] in label_dict:
            label_dict[y[i]].append(i)
        else: 
            label_dict[y[i]] = [i]
        
    return dict(OrderedDict(sorted(label_dict.items())))


def dirichlet_distribution(alpha, idx_by_label, size, no_change:int):
    """ Create Imbalanced data using dirichlet distribution """
    class_composition = np.array([len(idx_by_label[k]) for k in sorted(idx_by_label.keys())], np.int64)
    # print("Original Class composition: ", class_composition)
    
    if no_change == 1:
        dataset = []
        for v in idx_by_label.values():
            dataset += v
        return dataset
    distribution = np.random.dirichlet([alpha]*len(idx_by_label), size=())
    idx_by_label = idx_by_label.copy()
    
    
    #Group data by label
    for label in idx_by_label:
        class_size = math.ceil(size * distribution[label])
        if not class_size:
            class_size = 1 #min number to support A.2 assumption (BBSE ICML '18)
        indices = np.random.randint(0,
                                   len(idx_by_label[label]),
                                   size=(class_size, ))
        idx_by_label[label] = np.unique([idx_by_label[label][i] for i in indices]).tolist()
    
    class_composition = np.array([len(idx_by_label[k]) for k in sorted(idx_by_label.keys())], np.int64)
    # print("Shifted Class composition: ", class_composition)
        
    #Build new dataset of indices
    dataset = []
    for v in idx_by_label.values():
        dataset += v
    return dataset #shifted distribution


def get_distribution(labels):
    """ Returns the distribution of classes as ratios """
    dist = dict(Counter(labels))
    total_size = 0
    for key, value in dist.items():
        total_size += value
    
    for key in dist:
        dist[key] /= total_size
        
    return dict(OrderedDict(sorted(dist.items())))


def plot(y, indices, dist_type='Train', plot=False):
    if not plot:
        return
    
    ### Original Distribution
    plt.bar(x=np.unique(y), height=get_distribution(y).values())
    plt.title(dist_type + " Original Distribution")
    plt.xlabel("Class")
    plt.ylabel("PMF")
    plt.grid()
    plt.show()

    ### Shifted Distribution
    plt.bar(x=np.unique(y[indices]), height=get_distribution(y[indices]).values())
    plt.title(dist_type + " Shifted Distribution")
    plt.xlabel("Class label")
    plt.ylabel("PMF")
    plt.grid()
    plt.show()
    
    
# TODO: Port seperately
def train(data, epochs=500, epsilon=1e-5, print_st=False): #implement transforms
    """
    Train the model.
    Assumes access to global variable: loss function
    """
    X_train, y_train, X_test, y_test = data #extract info
    
    start_time = time.time()
    losses = []

    model = Network().to(device) #load local model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    # implement backprop
    loss_function = nn.CrossEntropyLoss()
    
    #gather accuracies
    train_accuracy = []
    test_accuracy = []
    
    for i in range(epochs):
        model.train() #set back to train
        y_pred = model(X_train)
        
        loss = loss_function(y_pred, y_train)
        losses.append(loss)
        
        ## training accuracy
        predictions = np.array(y_pred.argmax(axis=1), dtype=np.int16)
        score = accuracy_score(y_train, predictions)
        train_accuracy.append(score)
        
        ## test accuracy
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            predictions = np.array(y_pred.argmax(axis=1), dtype=np.int16)
            score = accuracy_score(y_test, predictions)
            test_accuracy.append(score)
        
        if loss.item() < epsilon:
            if print_st:
                print(f"Model Converged at epoch {i + 1}, loss = {loss.item()}")
            break
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if print_st:
        print(f"Total training time (sec): {time.time() - start_time}, loss - {loss.item()}")

    return model, losses, train_accuracy, test_accuracy


# TODO: Port seperately
def predict(model, data):
    """ Predict accuracy => y_hat = f(x). """
    X_test, y_test = data
    
    model.eval() #set to evaluation mode
    # predict X_test data
    predictions=[]
    with torch.no_grad():
        for i, data in enumerate(X_test):
            y_pred = model(data)
            predictions.append(y_pred.argmax().item())

    predictions = np.array(predictions, dtype=np.int16)
    score = accuracy_score(y_test, predictions)
    return score, predictions


def plot_cost(cost, train_acc, test_acc, title, to_plot):
    if not to_plot:
        return
    """ Plots accuracy + loss eval curve over number of epochs """
    plt.plot(cost, label='loss')
    plt.plot(train_acc, label='training accuracy')
    plt.plot(test_acc, label='test accuracy')
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
    
    
def calculate_confusion_matrix(X, Y, k, f:Network):
    """ 
    Calculates value for \hat{C}_{\hat{y}, y}
    @Params:
    - X : Validation data, i.e. X2
    - Y : Validation labels, i.e. Y2
    """
    conf_matrx = np.zeros(shape=(k, k))
    #freeze params
    f.eval()
    predictions=[]
    with torch.no_grad():
        for i, data in enumerate(X):
            y_pred = f(data)
            predictions.append(y_pred.argmax().item())
    
    predictions = np.array(predictions)
    for i in range(k):
        for j in range(k):    
            idxs = np.where((predictions == i) & (Y.numpy() == j))[0]
            conf_matrx[i, j] = float(len(idxs) / len(X))
    return conf_matrx, k

def calculate_target_priors(X, k, f):
    """ Calculates \hat{μ}_\hat{y} """
    preds = np.array([f(xp).argmax().item() for xp in X], np.int16)
    target_priors = np.zeros(k)
    for i in range(k):
        target_priors[i] = len(np.where(preds == i)[0]) / len(preds)
    return target_priors


def get_true_label_weights(y_train, k, y_test):
    """ 
    Returns true label weights based on source/medial and target dist. 
    @Param: 
    - y_train : training labels
    - k : number of classes
    - y_test : test labels
    """
    true_target_priors = np.zeros(k)
    true_source_priors = np.zeros(k)
    
    for i in range(k):
        true_target_priors[i] = float(len(np.where(y_test == i)[0])) / len(y_test)
        
    for i in range(k):
        true_source_priors[i] = float(len(np.where(y_train == i)[0])) / len(y_train)
    
    regularizer = 0.2
    try:
        optimal_weights = true_target_priors / true_source_priors #optimal weights
    except ZeroDivisionError:
        print(f"Assumption violated")
        optimal_weights = np.array([1/k]*k)
    return (1 - regularizer) + optimal_weights * regularizer

def compute_weights(cmf, target_priors, delta):
    """ Computes label weights """
    w, _ = np.linalg.eig(cmf + np.random.uniform(0, 1e-3, size=cmf.shape))
    #0 < delta < 1/k where k = number of classes.
    if abs(w.real.min()) <= delta: #non invertible matrix
        return np.full(shape=len(target_priors), fill_value=float(1 / len(target_priors)))
    
    try:
        label_weights = np.linalg.inv(cmf) @ target_priors
    except np.linalg.LinAlgError:
        label_weights = np.linalg.inv(cmf + np.random.uniform(0, 1e-3, size=cmf.shape)) @ target_priors
    
    label_weights = abs(label_weights)
    label_weights /= label_weights.sum()
    #label_weights[label_weights < 0] = 0 #strictly set rare occurances to 0 instead of abs (see BBSE)
    
    return label_weights


def train_iw(data, label_weights, network=None, epochs=500, print_st=True): #implement transforms
    """ Train model using class weights """
    X, y, X_test, y_test = data
    start_time = time.time()
    m, k = len(X), len(np.unique(y))
    
    loss_function = nn.CrossEntropyLoss(weight=torch.DoubleTensor(label_weights))
    
    losses = []
    
    model = Network().to(device) #load local model
    
    if network: 
        cloned_params = {}
        for layer in network.state_dict():
            cloned_params[layer] = network.state_dict()[layer].clone()
        
        model.load_state_dict(cloned_params)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    #gather accuracies
    train_accuracy = []
    test_accuracy = []
    
    for i in range(epochs):
        
        model.train() #set back to train
        
        y_pred = model(X)
        loss = loss_function(y_pred, y)
        losses.append(loss)
        
        ## training accuracy
        predictions = np.array(y_pred.argmax(axis=1), dtype=np.int16)
        score = accuracy_score(y, predictions)
        train_accuracy.append(score)
        
        ## test accuracy
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            predictions = np.array(y_pred.argmax(axis=1), dtype=np.int16)
            score = accuracy_score(y_test, predictions)
            test_accuracy.append(score)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if print_st:
        print(f"Total training time (sec): {time.time() - start_time}, loss - {loss.item()}")

    return model, losses, train_accuracy, test_accuracy


def predict_IW(model, label_weights, data):
    """ Predict accuracy => y_hat = f(x). Refer to BBSE, ICML '18 """
    X_test, y_test = data
    model.eval() #set to evaluation mode
    predictions=[]
    with torch.no_grad():
        for i, data in enumerate(X_test):
            y_pred = model(data)
            y_pred *= label_weights #IW softmax
            
            predictions.append(y_pred.argmax().item())

    predictions = np.array(predictions, dtype=np.int16)
    score = accuracy_score(y_test, predictions)
    return score, predictions



### ERM for IS

# https://github.com/Angie-Liu/labelshift/blob/5bbe517938f4e3f5bd14c2c105de973dcc2e0917/label_shift.py#L123
def compute_w_opt(C_yy, mu_y, mu_train_y, rho=1e-3):
    """
    Finds optimal weights as ERM problem.
    C_yy: conf_matrx ; output of `calculate_confusion_matrix`
    mu_y: output of `calculate_target_priors` for testing set
    mu_train_y : output of `calculate_target_priors` for training set
    """
    n = C_yy.shape[1]
    theta = cp.Variable(n)
    b = mu_y - mu_train_y
    objective = cp.Minimize(cp.pnorm(C_yy @ theta - b) + rho * cp.pnorm(theta))
    constraints = [-1 <= theta]
    prob = cp.Problem(objective, constraints)

    result = prob.solve()
    w = 1 + theta.value
    return w
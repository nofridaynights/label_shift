# MASIW - Meta Subsampling Importance Weighting
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

import tqdm
import math
import time
import argparse
from collections import Counter, deque, OrderedDict

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Network
from maml import MAML
from utils import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Enable cuda if available

def label_shift(args:dict):
    """ 
    Runs main label shift experiment for specific source and target distribution 
    @Returns:
    - RESULTS : 
    {
        'naive' : None, #default training
        'bbse'  : None, #IW only
        'oracle' : None, #use true label weights
        'malls' : None,  #subsampling + importance weighting
        'masiw' : None, #meta-learning + subsampling + importance weighting
    }
    """
    
    # set reproducibility
    if args.seed >= 0:
        np.random.seed(args.seed)
        _ = torch.manual_seed(args.seed)
        
    RESULTS = {}
        
    
    X, y = load_digits(return_X_y=True) #multiclassification
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.target_ratio, random_state=42)
    
    k_classes = len(np.unique(y_train)) #number of classes in training dist. assuming this equals test set
    
    
    #### --- Create Imbalanced Dataset
    
    idx_by_label = group_by_label(y_train) #label : [indices of all labels]
    
    #Source distribution shift
    size = X_train.shape[0]**2
    shifted_dist_idx = dirichlet_distribution(
        alpha=args.source_alpha, idx_by_label=idx_by_label, size=size, no_change=int(args.keep_source))
    
    #Test distribution shift
    idx_by_label = group_by_label(y_test) #label : [indices of all labels]
    size = X_test.shape[0]**2
    shifted_test_dist_idx = dirichlet_distribution(
        alpha=args.target_alpha, idx_by_label=idx_by_label, size=size, no_change=int(args.keep_target))
    
    #train Distribution shift
    plot(y_train, shifted_dist_idx, 'Train', args.display_plots)
    
    #test Distribution shift
    plot(y_test, shifted_test_dist_idx, 'Test', args.display_plots)
    
    #### --- Sync With Data

    ### No subsampling - take source Dist.
    X_train, y_train = X_train[shifted_dist_idx], y_train[shifted_dist_idx]

    ### Shifting test distribution
    X_test, y_test = X_test[shifted_test_dist_idx], y_test[shifted_test_dist_idx]
    
    #Get source (train) and target (test) label distributions
    dist_train = get_distribution(y_train)
    dist_test  = get_distribution(y_test)

    #### --- Train Naive Model
    
    ##typecast to tensors
    X_train = torch.DoubleTensor(X_train).to(device)
    X_test = torch.DoubleTensor(X_test).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    y_test = torch.LongTensor(y_test).to(device)
    
    
    model_normal, cost, training_accuracy, test_accuracy = train(
        (X_train, y_train, X_test, y_test), print_st=args.display_plots, epochs=args.epochs)
    
    #graph cost
    plot_cost(cost, training_accuracy, test_accuracy, 'Full Batch Training Cost', args.display_plots)
        
    #### --- Test Model
    
    ### Estimated distribution
    score, predictions = predict(model_normal, (X_test, y_test))
    
    RESULTS['naive'] = score
    
    if args.alg == 'naive':
        return RESULTS
    
    #### -- MALLS Subsampling
    
    # Generate Medial Distribution
    biased_probs = 1. / np.array(list(dist_train.values()))
    biased_probs /= np.sum(biased_probs)
    
    p = np.zeros(y_train.shape)

    for i in range(len(p)):
        p[i] = biased_probs[y_train[i]]

    p /= p.sum() #normalize
    medial_idx = np.random.choice(np.arange(len(y_train)), size=y_train.shape, replace=True, p=p)
    
    
    if args.display_plots:
        ### Medial Distribution
        plt.bar(
            x=np.unique(y_train[medial_idx]), height=get_distribution(y_train[medial_idx].numpy()).values())
        plt.title("Medial Distribution")
        plt.xlabel("Class label")
        plt.ylabel("PMF")
        plt.grid()
        plt.show()
    
    if args.alg != 'bbse' or args.alg != 'mlls':
        ### Subsampling - take Medial Dist.
        X_train, y_train = X_train[medial_idx], y_train[medial_idx]
    
    ### --- BBSE Label Shift
    data = X_train.clone(), y_train.clone() #store original training distribution.

    #Split training into training (source) and validation (hold-out)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=args.validation_ratio, random_state=42)
    
    ### obtain classifier by training on X_train, y_train
    f, cost, training_accuracy, test_accuracy = train(
        (X_train, y_train, X_test, y_test), print_st=args.display_plots, epochs=args.epochs)
    
    
    #graph cost
    plot_cost(cost, training_accuracy, test_accuracy, 'Source only Cost', args.display_plots)
    
    ### --- Generate Label Shift
    conf_matrix, k = calculate_confusion_matrix(X_validation, y_validation, k_classes, f)
    mu = calculate_target_priors(X_test, k, f)
    #generate label weights, if possible
    if args.alg == 'bbse' or args.alg == 'malls':
        label_weights = compute_weights(conf_matrix, mu, args.delta)
    elif args.alg == 'mlls' or args.alg == 'masiw': #MASIW and RLLS
        label_weights = (1 - args.lambda_rlls) + args.lambda_rlls * compute_w_opt(conf_matrix, mu, calculate_target_priors(X_train, k, f))
    
    #!!! report true label weights, if arg specified !!!
    if args.alg == 'oracle':
        label_weights = get_true_label_weights(y_train, k_classes, y_test)
    
    # print(f'label weights: {label_weights}')
    
    #### --- Importance Weighting Training
    X_train, y_train = data #regain data
    f_weighted, cost, training_accuracy, test_accuracy = train_iw(
        (X_train, y_train, X_test, y_test), label_weights, f, epochs=args.epochs, print_st=args.display_plots)
    
    plot_cost(cost, training_accuracy, test_accuracy, 'Full Source Training Cost', args.display_plots)
    
    ### --- Importance Weighting Test
    score, _ = predict_IW(f_weighted, label_weights, (X_test, y_test)) ### Prediction

    
    if args.alg != 'masiw':
        RESULTS[args.alg] = score
        return RESULTS

    #### --- MAML - Importance Weight Bias Reduction
    maml = MAML(X_validation, y_validation, f_weighted, label_weights) # declare maml

    for _ in range(args.meta_updates):
        maml.update()
    
    label_weights = maml.get_label_weights()   
    # f_meta_weighted, cost, training_accuracy, test_accuracy = train_iw(
    #     (X_train, y_train, X_test, y_test), label_weights, f_weighted, epochs=args.epochs, print_st=args.display_plots)
    score, _ = predict_IW(f_weighted, label_weights, (X_test, y_test))
    
    RESULTS['masiw'] = score
    
    return RESULTS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Efficient Label Shift based Domain Adaption')

    #source alpha
    parser.add_argument('-source_alpha', metavar='Dirichlet Distibution parameter', type=float, default=1,
                        help='Magnitude of Label Shift based on dirichlet dist.')
    #keep source alpha (1 := keep, 0 := use original dist)
    parser.add_argument('-keep_source', type=int, default=1,
                        help='Keep original distribution or dirichlet label shift simulation')
    #target distribution alpha
    parser.add_argument('-target_alpha', metavar='Dirichlet Distibution parameter', type=float, default=1,
                        help='Magnitude of Label Shift based on dirichlet dist.')
    #keep target alpha (1 := keep, 0 := use original dist)
    parser.add_argument('-keep_target', type=int, default=1,
                        help='Keep original distribution or dirichlet label shift simulation')
    #target distribution ratio
    parser.add_argument('-target_ratio', type=float, default=0.2,
                        help='Proportion of original data to be set aside for target set')
    #validation distribution ratio
    parser.add_argument('-validation_ratio', type=float, default=0.3,
                        help='Proportion of train data to be set aside for holdout set')
    #BBSE delta
    parser.add_argument('-delta', type=float, default=1e-9,
                        help='BBSE inverse confusion matrix threshold paramter, delta.')
    #display graphs
    parser.add_argument('-display_plots', type=bool, default=False,
                        help='Show graphs when running experiment?')
    #algorithm type
    parser.add_argument('-alg', metavar='Label Shift Algorithm', type=str, default='MASIW',
                        help='Type of Label Shift algorithm to run')
    #number of epochs to run algs
    parser.add_argument('-epochs', metavar='Gen epoch count', type=int, default=350,
                        help='Number of epochs to run all learning algorithms')
    #number of MAML updates
    parser.add_argument('-meta_updates', metavar='number of meta updates', type=int, default=3,
                        help='Number of MAML meta updates. Note: More updates increases sensitivity')
    #use true label weights
    parser.add_argument('-use_oracle', metavar='True label weights for IW', type=int, default=0,
                        help='Query labels from test dist to compute true IW')
    #RLLS regularizer (without cp)
    parser.add_argument('-lambda_rlls', metavar='Regularizer for RLLS', type=float, default=1,
                        help='lambda for RLLS')
    #set seed
    parser.add_argument('-seed', metavar='Set seed value', type=int, default=-1,
                        help='Int >= 0 for seed reproducibility. Set as -1 (default) for no seed.')
    
    args = parser.parse_args() #parse arguments
    
    #Run MASIW
    # result = label_shift(args)
    
    # print(result[args.alg])
    
    # temp - Run all algorithms
    algs = ['oracle', 'naive', 'bbse', 'mlls', 'malls', 'masiw']
    for alg in algs:
        args.alg = alg
        result = label_shift(args)
        print(result[alg])
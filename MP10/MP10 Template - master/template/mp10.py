# mp9.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Modified by James Soole for the Fall 2023 semester

import sys
import argparse
import configparser
import copy

import numpy as np
import torch
import pickle
import reader
import neuralnet_part1 as p_part1
import neuralnet_part2 as p_part2
import neuralnet_leaderboard as p_leaderboard
from utils import compute_accuracies, get_parameter_counts

"""
This file contains the main application that is run for this MP.
"""

def main(args):
    reader.init_seeds(args.seed)

    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(args.dataset_file)
    train_set       = torch.tensor(train_set, dtype=torch.float32)
    train_labels    = torch.tensor(train_labels, dtype=torch.int64)
    dev_set         = torch.tensor(dev_set, dtype=torch.float32)

    if(args.leaderboard):
        p = p_leaderboard
    elif(args.part1):
        p = p_part1
    elif(args.part2):
        p = p_part2
    else:
        print("Please specify a model to run, e.g. --part1, --part2, or --leaderboard.")
        return

    L, predicted_labels, net = p.fit(train_set, train_labels, dev_set, args.epochs)

    if args.leaderboard:
        torch.save(net, "net.model")
        torch.save(net.state_dict(), 'state_dict.state')

    l = net.step(train_set[-1,:].unsqueeze(0), train_labels[-1].unsqueeze(0))

    assert type(l) == float, "your step function returned the loss as {} instead of a scalar of type float. Make sure to use .detach().cpu().numpy() on the loss before you return it in step function and to convert the output to a python float!".format(type(l))
    assert type(predicted_labels) == np.ndarray, "your fit function returned the predicted labels as {} instead of np.ndarray. Make sure to use .detach().cpu().numpy() on the network output - and don't forget to argmax it!".format(type(predicted_labels))
    assert type(L) == list,"your fit function returned the losses as {} instead of list. Make sure you are returning a list of losses (with length equal to the number of epochs)".format(type(L))
    
    accuracy, conf_m = compute_accuracies(predicted_labels, dev_set, dev_labels)
    num_parameters, params = get_parameter_counts(net)
    print("\n Accuracy: ", accuracy)
    print("\n Confusion Matrix = \n {}".format(conf_m))
    print('\n Parameters = {} \n'.format(num_parameters))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440/ECE448 MP10: Neural Nets and PyTorch')

    parser.add_argument('--dataset', dest='dataset_file', type=str, default = './data/mp10_data',
                        help='directory containing the training data')
    parser.add_argument('--leaderboard', dest="leaderboard", action='store_true', 
                        help='If set, saves net.model and state_dict.state for the leaderboard.')
    parser.add_argument('--part1', dest="part1", action='store_true',
                        help='If set, runs the model defined in neuralnet_part1.py')
    parser.add_argument('--part2', dest="part2", action='store_true',
                        help='If set, runs the model defined in neuralnet_part2.py')
    parser.add_argument('--epochs',dest="epochs", type=int, default = 50,
                        help='Training Epochs: default 50')
    parser.add_argument('--seed', dest="seed", type=int, default=42,
                        help='seed source for randomness')

    args = parser.parse_args()
    main(args)

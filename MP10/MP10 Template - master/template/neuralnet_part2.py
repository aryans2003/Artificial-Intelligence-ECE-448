# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10 Part2. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader, TensorDataset # add TensorDataset


class NeuralNet(nn.Module): # modified from Part 1
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        Parameters:
        lrate (float): Learning rate for the model.
        loss_fn (callable): A loss function defined as follows:
            Parameters:
                yhat (Tensor): An (N, out_size) Tensor.
                y (Tensor): An (N,) Tensor.
            Returns:
                Tensor: A scalar Tensor that is the mean loss.
        in_size (int): Input dimension.
        out_size (int): Output dimension.
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        # For Part 1, the network should have the following architecture (in terms of hidden units):
        # in_size -> h -> out_size, where 1 <= h <= 256


        # TODO Define the network architecture (layers) based on these specifications.
        
        # first, we want to define a convolutional neural network
        # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html, 'Define a Convolutional Neural Network'
        # our new size is (batch_size, channel_num, height, width) instead of (batch_size, 31*31*3)
        # according to the compatible with a CNN layer part...
        # channel_num = 3
        # height = 31
        # width = 31
        # kernel size always 3
        # padding always 1
        # dilation always 1
        # stride always 1
        # using LeakyReLU for activation function since it was part of our quiz skills so must be the best method
        self.model = nn.Sequential(
            # first convolution layer --> activation function --> first pooling layer
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
            # in_channels = 3 (corresponds to RGB)
            # out_channels = for output channels, in the doc it ascends 6, 16... had to play around w diff values to match accuracy and params
            nn.Conv2d(in_channels = 3, out_channels = 30, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.LeakyReLU(), # activation function between layers
            # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
            # just using the same values they give for this...
            nn.MaxPool2d(2, 2),
            
            # second convolution layer --> activation function --> second pooling layer
            # subsequent in_channels have to be equal to the out_channels from the previous convolution layer
            # increase out_channels
            # rest same
            nn.Conv2d(in_channels = 30, out_channels = 60, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.LeakyReLU(), # activation function between layers
            nn.MaxPool2d(2, 2), # just using the same values they give for this...
            
            # third convolution layer --> activation function --> third pooling layer
            # subsequent in_channels have to be equal to the out_channels from the previous convolution layer
            # increase out_channels
            # rest same
            nn.Conv2d(in_channels = 60, out_channels = 90, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.LeakyReLU(), # activation function between layers
            nn.MaxPool2d(2, 2), # just using the same values they give for this...
            
            # flatten output for fully connected layers
            nn.Flatten(),
            
            # create the fully connected layers using Linear object as we did before
            nn.Linear(90 * 3 * 3, 500), # 90 channels, 3x3 spatial grid, had to play around w the 500 value for hidden states started w/ 170
            nn.LeakyReLU(), # activation function between layers
            # output layer
            nn.Linear(500, out_size) # using output size for output layer, had to play around w the 500 value for hidden states started w/ 170
        )
          
        # initializing an optimizer object in this function to optimize network in step()
        # using SGD because it works well for smaller networks and is simplest
        # using our given lrate as well as the standard momentum of 0.9 in the torch.optim docu
        # we need to give optim an iterable with our parameters
        self.optimizer = optim.SGD(self.parameters(), lr = lrate, momentum = 0.9)
        
    def forward(self, x): # same as Part 1 except resizing
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        Parameters:
        x (Tensor): An (N, in_size) Tensor.

        Returns:
        Tensor: An (N, out_size) Tensor of output from the network.
        """
        # TODO Implement the forward pass.
        # since we defined as a sequential object we just simply call the model
        # but we need to reshape the input for part 2 to match...
        # batch size --> x.size(0), 3, 31, 31
        return self.model(x.view(x.size(0), 3, 31, 31))

    def step(self, x, y): # same as Part 1
        """
        Performs one gradient step through a batch of data x with labels y.

        Parameters:
        x (Tensor): An (N, in_size) Tensor representing the input data.
        y (Tensor): An (N,) Tensor representing the labels.

        Returns:
        float: The total empirical risk (mean of losses) for this batch.
        """
    
        # perform the gradient update thru one batch of training data
        
        # let's use the optimizer object that I initialized
        # be sure to call zero_grad() on optimizer to clear the gradient buffer
        self.optimizer.zero_grad()
        
        # after this, we compute the forward pass
        forward_pass = self.forward(x)
        
        # compute the loss in between using our output from forward pass and target y
        loss_value = self.loss_fn(forward_pass, y)
        
        # compute backward pass on the loss value
        loss_value.backward()
        
        # just call the built-in step() function
        self.optimizer.step()
        
    
        # Important, detach and move to cpu before converting to numpy and then to python float.
        # Or just use .item() to convert to python float. It will automatically detach and move to cpu.
        # return loss_value.item()
        return loss_value.item()


def fit(train_set,train_labels,dev_set,epochs,batch_size=100): # same as Part 1
    """
    Creates and trains a NeuralNet object 'net'. Use net.step() to train the neural net
    and net(x) to evaluate the neural net.

    Parameters:
    train_set (Tensor): An (N, in_size) Tensor representing the training data.
    train_labels (Tensor): An (N,) Tensor representing the training labels.
    dev_set (Tensor): An (M,) Tensor representing the development set.
    epochs (int): The number of training epochs.
    batch_size (int, optional): The size of each training batch. Defaults to 100.

    This method must work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values if your initial choice does not work well.
    For Part 1, we recommend setting the learning rate to 0.01.

    Returns:
    list: A list of floats containing the total loss for every epoch.
        Ensure that len(losses) == epochs.
    numpy.ndarray: An (M,) NumPy array (dtype=np.int64) of estimated class labels (0,1,2, or 3) for the development set (model predictions).
    NeuralNet: A NeuralNet object.
    """
    
    # inputs -- batch size, training epochs
    
    # 1) fit() should construct a NeuralNet object
    
    # learning rate to 0.01
    # use loss_fn as CrossEntropyLoss, https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    # use train_set.shape[1] to get features in dataset --> input layer size
    # the unique amount of training labels should be our output layer size
    net = NeuralNet(lrate = 0.01, loss_fn = nn.CrossEntropyLoss(), in_size = train_set.shape[1], out_size = len(torch.unique(train_labels)))
    
    # 2) and then iteratively call the neural net's step() function to train the network
    
    # Standardize the training set
    # subtract set from the mean and divide the difference by the standard deviation
    train_set_standardized = (train_set - torch.mean(train_set)) / torch.std(train_set)
    
    # since we want to train by batch, we can implement a dataloader, 
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html "preparing your data for training with DataLoaders"
    # module defined in given imports so looks like we're on the right track
    # create TensorDataset to combine features and labels in training set
    train_dataset = TensorDataset(train_set_standardized, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
    
    losses = list() # init list to store losses for each epoch for return
    
    # for each epoch of training....
    for epoch in range(epochs):
        epoch_loss = 0 # set up our count for total losses in one epoch
        # loop thru one batch's features and labels from our DataLoader for training set
        for batch in train_dataloader: 
            # Use net.step() to train the neural net, outputs the specific batch's loss
            batch_loss = net.step(batch[0], batch[1])
            # append the current batch loss to the total loss for the epoch
            epoch_loss += float(batch_loss)
        # compute average loss for the epoch... we can get the total batches in one epoch from the dataloader
        average_loss = epoch_loss / len(train_dataloader)
        # append it to our list of all losses for each epoch
        # use float to ensure we're storing Python floats
        # part of the detach losses step....
        losses.append(float(average_loss))
        # 3) run the neural net on the dev set
        # from https://pytorch.org/tutorials/beginner/basics/data_tutorial.html, we also read the "Per-Epoch Activity" section for the performing validation part
        with torch.no_grad():
            # use net(x) to evaluate the neural net
            dev_evaluate = net(dev_set)
            # getting estimated class labels for the dev set
            # https://pytorch.org/docs/stable/generated/torch.argmax.html
            # input tensor is dev_evaluate
            # reduce dimension 1
            # store as np arr
            dev_predictions = torch.argmax(dev_evaluate, dim = 1).numpy()

    # return a list of the losses for each epoch of training
    # return np arr with estimated class labels for the dev set
    # return trained NeuralNet
    return losses, dev_predictions, net
    
    # Important, don't forget to detach losses and model predictions and convert them to the right return types.
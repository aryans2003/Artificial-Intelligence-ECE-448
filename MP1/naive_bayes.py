# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter
from collections import defaultdict # to initialize counts for dicts to 0
# you should also import nltk, numpy, (tqdm) already imported
import nltk
import numpy as np


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naive_bayes(train_set, train_labels, dev_set, laplace=1.0, pos_prior=0.5, silently=False):
    print_values(laplace,pos_prior)

    # Training Phase

    # we use the Counter library to hold the positive, negative words, a set to hold all words, a list to hold predicted class labels
    yhats = []
    positive_words = Counter()
    negative_words = Counter()
    all_words = set()

    # we can use zip() to iterate thru both the set and label lists simultaneously
    for(review, label) in zip(train_set, train_labels): # extract review, label from training set, training labels
        for word in review: # extract each word from the review
            all_words.add(word) # add the word into the set
            if label == 1: # label is '1', we have a positive word
                    positive_words[word] += 1 # increment the count of the existing word in the dictionary
            elif label == 0: # if label is '0', we have a negative word
                    negative_words[word] += 1 # increment the count of the existing word in the dictionary
    unique_words = len(all_words) # calculate the number of unique words, size of vocabulary

    # calculating the number of values in the positive and negative dictionaries
    positive_words_count = 0
    for count in positive_words.values():
         positive_words_count += count
    negative_words_count = 0
    for count in negative_words.values():
         negative_words_count += count

    # development phase

    # want to calculate the probability of each review in the development set
    for review in tqdm(dev_set, disable=silently): # using tqdm to loop thru the dev_set for speed, use the silently var in the header for the progress bar display
        # calculating prior probabilities given the input parameters (cannot hardcode)
        prior_probability_positive = math.log(pos_prior)
        prior_probability_negative = math.log(1 - pos_prior)
        for word in review: # extract each word from the review
             # Laplace smoothing formula:
             # apply log to convert multiplication to addition
             # numerator: each word's count + laplace smoothing factor to handle zero probabilities
             # denominator: all the word's count + (laplace * (unique # of words + 1 to account for 0s))
             prior_probability_positive += math.log((positive_words[word] + laplace) / (positive_words_count + laplace * (unique_words+1)))
             prior_probability_negative += math.log((negative_words[word] + laplace) / (negative_words_count + laplace * (unique_words+1)))
        # append 1 if prior_probability_positive > prior_probability negative, else 0 (based on higher probability pretty much)
        yhats.append(1 if prior_probability_positive > prior_probability_negative else 0)
    return yhats
# bigram_naive_bayes.py
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


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
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
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=0.05, bigram_laplace=1.0, bigram_lambda=0.5, pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    # Training Phase

    # we use the Counter library to hold the positive, negative words, a set to hold all words, a list to hold predicted class labels
    yhats = []
    positive_words = Counter()
    negative_words = Counter()
    all_words = set()

    # we want to add the same data parameters as above for bigrams, above were for 'unigrams'
    positive_bigrams = Counter()
    negative_bigrams = Counter()
    all_bigrams = set()

    # we can use zip() to iterate thru both the set and label lists simultaneously
    for(review, label) in zip(train_set, train_labels): # extract review, label from training set, training labels
        for word in review: # extract each word from the review
            all_words.add(word) # add the word into the set
            if label == 1: # label is '1', we have a positive word
                    positive_words[word] += 1 # increment the count of the existing word in the Counter class
            elif label == 0: # if label is '0', we have a negative word
                    negative_words[word] += 1 # increment the count of the existing word in the Counter class
        for i in range(len(review) - 1): # loop thru the review words, we loop - 1 because we check in pairs (curr word then one before it)
            bigram = review[i] + " " + review[i + 1] # isolate each bigram by getting the curr review word, a space, and the next review word
            all_bigrams.add(bigram) # add the bigram to the set of all bigrams
            if label == 1: # if label is '1', we have a psotive bigram
                 positive_bigrams[bigram] += 1 # increment the count of the existing bigram in the Counter class
            elif label == 0: # if label is '0', we have a negative bigram
                 negative_bigrams[bigram] += 1 # incremeent the count of the existing bigram in the Counter class

    unique_words = len(all_words) # calculate the number of unique words, size of vocabulary
    unique_bigrams = len(all_bigrams) # calculate the number of unique bigrams, size of vocabulary

    # calculating the number of values in the positive and negative Counter class for unigrams
    positive_words_count = 0
    for count in positive_words.values():
         positive_words_count += count
    negative_words_count = 0
    for count in negative_words.values():
         negative_words_count += count

    # calculating the number of values in the positive and negative Counter class for bigrams
    positive_bigrams_count = 0
    for count in positive_bigrams.values():
        positive_bigrams_count += count
    negative_bigrams_count = 0
    for count in negative_bigrams.values():
         negative_bigrams_count += count

    # development phase

    # want to calculate the probability of each review in the development set
    for review in tqdm(dev_set, disable=silently): # using tqdm to loop thru the dev_set for speed, use the silently var in the header for the progress bar display

        # -- for unigrams (old code)
        # calculating prior probabilities given the input parameters (cannot hardcode) 
        prior_probability_positive = math.log(pos_prior)
        prior_probability_negative = math.log(1 - pos_prior)

        # -- for bigrams (new code)
        # set new values for bigrams, equal to old values but we can't use them because we are updating them only for bigrams
        prior_probability_positive_bi = prior_probability_positive
        prior_probability_negative_bi = prior_probability_negative

        for word in review: # extract each word from the review
             
             # -- for unigrams (old code)
             # Laplace smoothing formula:
             # apply log to convert multiplication to addition
             # numerator: each word's count + laplace smoothing factor to handle zero probabilities
             # denominator: all the word's count + (laplace * (unique # of words + 1 to account for 0s))
             prior_probability_positive += math.log((positive_words[word] + unigram_laplace) / (positive_words_count + unigram_laplace * (unique_words+1)))
             prior_probability_negative += math.log((negative_words[word] + unigram_laplace) / (negative_words_count + unigram_laplace * (unique_words+1)))

        # -- for bigrams (new code)
        for i in range(len(review) - 1): # loop thru the review words, we loop - 1 because we check in pairs (curr word then one before it)
            bigram = review[i] + " " + review[i + 1] # isolate each bigram by getting the curr review word, a space, and the next review word
            # pretty much use the same formula as before
            prior_probability_positive_bi += math.log((positive_bigrams[bigram] + bigram_laplace) / (positive_bigrams_count + bigram_laplace * (unique_bigrams + 1)))
            prior_probability_negative_bi += math.log((negative_bigrams[bigram] + bigram_laplace) / (negative_bigrams_count + bigram_laplace * (unique_bigrams + 1)))

        # calculated weighted sum for bigram Counters
        weighted_sum_bi_pos = bigram_lambda * prior_probability_positive_bi
        weighted_sum_bi_neg = bigram_lambda * prior_probability_negative_bi

        # calculated weighted sum for unigram Counters (incldue bigram_lambda)
        # the bigram lambda controls how much weight the bigram probabilities contribute to the combined set compared to the unigram probabilities
        weighted_sum_uni_pos = (1 - bigram_lambda) * prior_probability_positive
        weighted_sum_uni_neg = (1 - bigram_lambda) * prior_probability_negative

        # combine these values to get our combined prior probabilities
        combined_prior_probabilities_positive = weighted_sum_bi_pos + weighted_sum_uni_pos
        combined_prior_probabilities_negative = weighted_sum_bi_neg + weighted_sum_uni_neg

        # append 1 if combined_prior_probability_positive > combined_prior_probability negative, else 0 (based on higher probability pretty much, taking into account unigrams and bigrams)
        yhats.append(1 if combined_prior_probabilities_positive > combined_prior_probabilities_negative else 0)

    return yhats # return
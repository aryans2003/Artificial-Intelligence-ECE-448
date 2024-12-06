"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-8   # exact setting seems to have little or no effect



def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences: 
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}

    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.

    # init function-scope vars
    word_freq = Counter() # dict to count occurrences of words for each tag
    word_tag = Counter() # dict to count word, tag pairs
    tag_freq = Counter() # dict to count occurrences for each tag
    unique_words = defaultdict(set) # set for unique words from each tag
    tag_pair = defaultdict(Counter) # NESTED dict for transitions
    total_sentences = len(sentences) # var to store total # of sentences

    # 1) computing initial tag probabilities
    # formula is --> (first_tag in sentence) / (total sentences)
    # loop thru each sentence
    for sentence in sentences:
        # extract first tag
        first_tag = sentence[0][1]
        # compute initial probability from the aforementioned formula
        init_prob[first_tag] += (1 / total_sentences)

    # 2) computing emission and transition probabilities together
    # loop thru sentences
    for sentence in sentences:
        for i, (word, tag) in enumerate(sentence):
            # increment of word, tag pair in our defined dict
            word_tag[(word, tag)] += 1
            # increment tag counter
            tag_freq[tag] += 1
            # increment word counter
            word_freq[tag] += 1
            # increment unique word counter --> use .add b/c set
            unique_words[tag].add(word)
            # for the transition counts...
            # want to check the bounds b/c we want current tag vs next tag
            if i < len(sentence) - 1:
                # compute next tag
                next_tag = sentence[i + 1][1]
                # increment tag pair with curr tag and next tag
                tag_pair[tag][next_tag] += 1

    # 3) calculating emission probabilities here
    # using the formula --> (word tag pairs + emit_epsilon) / (# of tags + emit_epsilon * unique words in tag t + 1)
    for (word, tag), count in word_tag.items():
        emit_prob[tag][word] = (count + emit_epsilon) / (tag_freq[tag] + emit_epsilon * (len(unique_words[tag]) + 1))
    # for UNK words in each tag
    for tag in word_freq:
        emit_prob[tag]['UNKNOWN'] = emit_epsilon / (tag_freq[tag] + emit_epsilon * (len(unique_words[tag]) + 1))

    # 4) computing transition probabilities
    # using the formula --> (transitions from ti to tj + epsilon_for_pt) / (# of tags ti + epsilon_for_pt * # unique tags after ti + 1)
    # loop thru tags
    for tag1, next_tags in tag_pair.items():
        # pre-compute denominator
        denominator = tag_freq[tag1] + epsilon_for_pt * (len(next_tags) + 1)
        for tag2, count in next_tags.items():
            trans_prob[tag1][tag2] = (count + epsilon_for_pt) / denominator
        # for UNKNOWN transitions
        trans_prob[tag1]['UNKNOWN'] = epsilon_for_pt / denominator

    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.

    # handle special case for i = 0
    # if i = 0, no transition probabilities are used cos there are no transitions
    # directly compute emission prob for each tag and store it as init log prob
    # init log_prob[tag_curr] from emission probabilities, and then set predict_tag_seq[tag_curr] to [tag_curr]
    if i == 0:
        # for each tag in emission probabilities
        for tag, tag_emit_prob in emit_prob.items():
            # calculate emission
            emission = tag_emit_prob.get(word, tag_emit_prob.get('UNKNOWN', emit_epsilon))
            # calculate log probability
            log_prob[tag] = prev_prob[tag] + math.log(emission)
            predict_tag_seq[tag] = [tag]
    # moving onto i != 0
    else:
        # loop thru tags in emission probabilities
        for curr_tag, curr_tag_emit_prob in emit_prob.items():
            # calculate emission
            emission = curr_tag_emit_prob.get(word, curr_tag_emit_prob.get('UNKNOWN', epsilon_for_pt))
            # initialize array to store probabilities for previous tags
            probabilities = []
            # loop thru previous tags from previous probabilities
            for prev_tag in prev_prob:
                # compute transition
                transition = trans_prob[prev_tag].get(curr_tag, trans_prob[prev_tag].get('UNKNOWN', epsilon_for_pt))
                # compute TOTAL probability --> prob of prev tag, add log of trans_prof, add log emission_prob
                total_prob = prev_prob[prev_tag] + math.log(transition) + math.log(emission)
                # add total prob and prev tag to arr of probs
                probabilities.append((total_prob, prev_tag))
            # use max() to find max prob and best previous tag, swap values
            max_prob, best_previous_tag = max(probabilities)
            # calculate log prob from the current tag
            log_prob[curr_tag] = max_prob
            # append curr_tag from best previous tag to predict_tag_seq
            predict_tag_seq[curr_tag] = prev_predict_tag_seq[best_previous_tag] + [curr_tag]
    # return results          
    return log_prob, predict_tag_seq

def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.

        # compute best last tag from log prob
        blt = max(log_prob, key = log_prob.get)

        # compute best tag sequence from best laast tag
        bts = predict_tag_seq[blt]

        # pair each word into temp_predicts list
        temp_predicts = list(zip(sentence, bts))

        # append to REAL PREDICTS LIST
        predicts.append(temp_predicts)
        
    # return result
    return predicts
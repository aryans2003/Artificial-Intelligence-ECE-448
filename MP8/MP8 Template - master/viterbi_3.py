"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5  # exact setting seems to have little or no effect

# first, write a helper function to classify a word into one of six types
def classify_word(word): # input should be a word
        # if starts and ends with a numerical digit
        if word[0].isdigit() and word[-1].isdigit():
             return "NUMERICAL"
        # if word is very short (1-3 chars)
        elif len(word) <= 3:
             return "VERY_SHORT"
        # if word is short (4-9 chars)
        elif 4 <= len(word) <= 9:
             # ends in -s
             if word[-1] == 's':
                  return "SHORT_PLURAL"
             # if word is any other letter
             else:
                  return "SHORT"
        # if word is long (at least 10 chars)
        elif len(word) >= 10:
             # ends in -s
             if word[-1] == 's':
                  return "LONG_PLURAL"
             # if word is any other letter
             else:
                  return "LONG"
        # outputs are all string labels in ALL CAPS


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences: 
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    smoothing_alpha = defaultdict(lambda: defaultdict(lambda: 0)) # create one for the smoothing alpha factor

    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.

    # init function-scope vars
    word_freq = Counter() # dict to count occurrences of words for each tag
    word_tag = Counter() # dict to count word, tag pairs
    tag_freq = Counter() # dict to count occurrences for each tag
    tag_trans_count = Counter() # dict to count transition probabilities for a tag
    total_transitions = 0 # var to hold occcurrences of transition probabilities
    unique_words = defaultdict(set) # set for unique words from each tag
    hapax_words = set() # set to hold hapax words
    hapax_count = 0 # var to count occurrences for hapax words (all)
    hapax_tag_type_pair = defaultdict(Counter) # NESTED dict to count tag-type pairs
    hapax_tag_type_new = defaultdict(Counter) # NESTED dict to store adjusted tag type counts
    hapax_new_total = 0 # var to count occurrences for hapax words after being updated (all)
    seen_alphas = {} # set to hold seen alpha tags

    # 1) parsing data and storing values in correct counters / dicts
    # loop thru sentences
    for sentence in sentences:
        for i, (word, tag) in enumerate(sentence):
            # increment of word, tag pair in our defined dict
            word_tag[(word, tag)] += 1
            # increment tag counter
            tag_freq[tag] += 1
            # increment word counter
            word_freq[word] += 1
            # increment unique word counter --> use .add b/c set
            unique_words[tag].add(word)
            
            # compute initial tag probabilities
            if i == 1:
                 init_prob[tag] += 1

            # compute transition counts after we've seen i == 1
            if i > 0:
                 # compute prev tag as index of i - 1 at position [1]
                 prev_tag = sentence[i - 1][1]
                 # increment count of prev_tag and tag together
                 tag_trans_count[(prev_tag, tag)] += 1

    # 2) compute hapax words
    # loop thru all the words and the amount of times they have been seen in word frequencies
    for word, count in word_freq.items():
            # if the word has only been seen once
            if count == 1:
                 # store the word into hapax_words
                 hapax_words.add(word)

    # 3) count tag-type pairs
    # loop thru words and tags in our stored dicts of words and tags
    for word, tag in word_tag:
            # if the word is hapax
            if word in hapax_words:
                 # get the word type
                 word_type = classify_word(word)
                 # increment the occurrence in our dict by tag and type
                 hapax_tag_type_pair[tag][word_type] += 1
                 # increment count of all hapax words 
                 hapax_count += 1

    # compute updated hapax counts and smoothing alpha
    # we want to update it b/c we dont want any hapax count to be 0 --> unseen probabilities

    # 4) hapax counts
    # loop thru tags
    word_types = ['NUMERICAL', 'VERY_SHORT', 'SHORT_PLURAL', 'SHORT', 'LONG_PLURAL', 'LONG']
    for tag in tag_freq:
            # loop thru the word types that we stored
            for word_type in word_types:
                # set a temp count equal to the occurrences of word_type for the given tag, if not present assign 0 to count
                temp_count = hapax_tag_type_pair[tag].get(word_type, 0)
                # if the count is still 0 after it is all over...
                if temp_count == 0:
                     # set it to 1 to avoid unseen probabilities
                     temp_count = 1
                # finally, set the temporary count back into our new updated dict
                hapax_tag_type_new[tag][word_type] = temp_count

    # loop thru the counts of the hapax values for each tag
    for counts in hapax_tag_type_new.values():
            # sum counts for current tag
            tag_total = sum(counts.values())
            # add curr tag total to overall total, now it contains the sum of all hapax counts across all tags
            hapax_new_total += tag_total

    # 5) smoothing alpha
    # loop thru tags
    for tag in tag_freq:
            # loop thru word types
            for word_type in hapax_tag_type_new[tag]:
                # set count equal to the amount of tag-type pairs based on the tag and word type
                count = hapax_tag_type_new[tag][word_type]
                # compute our smoothing alpha for the given tag, type
                smoothing_alpha[tag][word_type] = emit_epsilon * (count / hapax_new_total)
    # loop thru tags
    for tag in tag_freq:
        # get dict of smoothing_alpha values for curr tag
        curr_alpha_val = smoothing_alpha[tag]
        # sum values in the current alpha value to get total alphas
        total_alphas = sum(curr_alpha_val.values())
        # store sum in a seen dict with the curr tag as the key, now seen alphas has sum of smoothing values for each tag
        seen_alphas[tag] = total_alphas
    
    # now all data done

    # 5) calculating init probabilities
    # set the total initial probabilities to the current sum of the initial probability values
    total_init_prob = sum(init_prob.values())
    # for each tag
    for tag in init_prob:
            # simply divide the init prob for that tag by the total initial probabilities
            # formula: (init prob for curr tag) / (total init prob)
            init_prob[tag] /= total_init_prob

    # 6) computing emission probabilities
    # for each tag
    for tag in tag_freq:
            # our vocabulary is all the unique words for that tag
            curr_V = unique_words[tag]
            # our alpha is the alpha that we saw for that tag
            curr_alpha = seen_alphas[tag]
            # for each word in our vocabular
            for word in curr_V:
                 # get the count of each word, tag pair
                 count = word_tag[(word, tag)]
                 # compute emission probability
                 # formula: word, tag pair / (freq of tag + smoothing alpha)
                 emit_prob[tag][word] = count / (tag_freq[tag] + curr_alpha)
            # for each type in tag's smoothing alpha
            for word_type in smoothing_alpha[tag]:
                 # compute unknown alpha
                 alpha_unk = smoothing_alpha[tag][word_type]
                 # use same formula for unknown_word type emission probability
                 emit_prob[tag][f"UNK_{word_type}"] = alpha_unk / (tag_freq[tag] + curr_alpha)

    # 7) computing transition probabilities
    # create our tag set from the tag freq dict we created
    tag_set = set(tag_freq.keys())
    # number of total tags is simply the length of that set
    num_tags = len(tag_set)
    # loop thru tags in the set
    for curr_tag in tag_set:
        # init curr_tag as defaultdict(float) which automatically initializes unseen transitions to 0
        trans_prob[curr_tag] = defaultdict(float)
        # calculate total transitions as the sum of the counts of transitions from the current tag
        # to each possible next tag, including the 'END' tag, using the tag transition count dictionary.
        total_transitions = sum(tag_trans_count.get((curr_tag, tag_end), 0) for tag_end in tag_set.union({'END'}))
        # loop thru tags w/ END tag
        for tag_end in tag_set.union({'END'}):
            # calculate count from our dict, 0 if not found
            count = tag_trans_count.get((curr_tag, tag_end), 0)
            # update transition prob for the tag and ending tag
            # formula: (count + epsilon) / (total transition prob + epsilon * (number of tags + 1 --> unseen tags))
            trans_prob[curr_tag][tag_end] = (count + epsilon_for_pt) / (total_transitions + epsilon_for_pt * (num_tags + 1))
        # calculate for unknown tag ends, same formula exc w/o count
        trans_prob[curr_tag]['UNK'] = epsilon_for_pt / (total_transitions + epsilon_for_pt * (num_tags + 1))

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

    word_type = classify_word(word) # get word type from our helper function

    # iterate thru tag and its emission probabilities in the passed emit_prob
    for curr_tag, curr_emit_prob in emit_prob.items():
        # set emission prob for the current word, or unknown word type if not found
        emission = curr_emit_prob.get(word) or curr_emit_prob.get(f"UNK_{word_type}", 0)
        # if emission is zero skip it and keep going
        if emission == 0:
            continue
        
        # calculate log of emission prob
        log_emission = log(emission)
        # get transition prob for curr tag
        curr_trans_prob = trans_prob[curr_tag]
        
        # set our max prob to negative inf, as per Piazza
        # will be updated as we see higher probs
        max_prob = float('-inf')
        # same as above, set best prev tag to None until we find one
        best_prev_tag = None

        # loop thru prev tag and its prob
        for prev_tag, prev_tag_prob in prev_prob.items():
            # get transition prob from prev tag to curr tag
            trans = curr_trans_prob.get(prev_tag, curr_trans_prob['UNK'])
            # if zero, skip
            if trans == 0:
                continue
            # calculate the probability of the tag
            prob = prev_tag_prob + log(trans) + log_emission
            # if higher than max prob
            if prob > max_prob:
                # swap it as the new max prob
                max_prob = prob
                # update best prev tag as this prev tag
                best_prev_tag = prev_tag

        # if we found a best prev tag
        if best_prev_tag is not None:
            # store the max prob for this curr tag
            log_prob[curr_tag] = max_prob
            # store best tag sequence for curr tag
            predict_tag_seq[curr_tag] = prev_predict_tag_seq[best_prev_tag] + [curr_tag]

    return log_prob, predict_tag_seq

def viterbi_3(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # train the model to get probabilities as in 1/2
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    # init list to store predictions, as in 1/2
    predicts = []

    # loop thru test sequences
    for sequence in test:
        # get curr sentence (remove 'START' and 'END')
        # START is at 1
        # END is at -1
        curr_sentence = sequence[1:-1]
        # the length of the sequence is the length of the sentence
        seq_len = len(curr_sentence)
        # if empty...
        if seq_len == 0:
            # append the start and end tokens 
            predicts.append([('START', 'START'), ('END', 'END')])
            # and move onto next sequence
            continue

        # for this Viterbi algorithm, we are going to implement backtracking
        # Viterbi matrix
        V = [{}]
        # backtracking dict
        dict = {}

        # for the first word's case, its a bit different

        # get first word
        word1 = curr_sentence[0]
        # get its type
        word1_type = classify_word(word1)

        # iterate thru tags in emission probs
        for tag in emit_prob:
            # get initial prob for tag, if not just use the epsilon pt val
            curr_init_prob = init_prob.get(tag, epsilon_for_pt)
            # get emission prob for the word, use UNKNOWN_word type if not found
            curr_emit_prob = emit_prob[tag].get(word1, emit_prob[tag].get(f"UNK_{word1_type}", 0))
            # if emit_prob is 0,
            if curr_emit_prob == 0:
                continue # skip
            # calculate and store probability for curr tag
            # using log probabilities
            V[0][tag] = log(curr_init_prob) + log(curr_emit_prob)
            # initialize backtrack for the tag in dict
            dict[tag] = [tag]
         
        # the rest of the words follow this format

        # iterate thru rest of words in sentence (from 1 to rest of length)
        for j in range(1, seq_len):
            # add a new dict for Viterbi matrix
            V.append({})
            # init new bactrack dict
            new_dict = {}

            # get curr word
            curr_word = curr_sentence[j]
            # get its type
            curr_word_type = classify_word(curr_word)

            # iterate thru all tags
            for curr_tag in emit_prob:
                # init max prob and best previous tag
                # this is similar to how I did it in the stepforward algo
                max_prob = float('-inf') # use -inf as stated in Piazza
                best_previous_tag = None # mark None until we find one
                # Get emission probability for the current word and tag
                curr_emission_prob = emit_prob[curr_tag].get(curr_word, emit_prob[curr_tag].get(f"UNK_{curr_word_type}", 0))
                # if emission prob is zero....
                if curr_emission_prob == 0:
                    continue # skip
                # iterate thru possible prev tags
                for prev_tag in V[j - 1]:
                    # get trans prob from prev to curr
                    curr_trans_prob = trans_prob[prev_tag].get(curr_tag, trans_prob[prev_tag]['UNK'])
                    # if trans prob is zero...
                    if curr_trans_prob == 0:
                        continue # skip
                    # get path prob
                    # USE LOG PROB
                    path_prob = V[j-1][prev_tag] + log(curr_trans_prob) + log(curr_emission_prob)
                    # if this better than our max prob
                    if path_prob > max_prob:
                        # swap em
                        max_prob = path_prob
                        # and update our tag
                        best_previous_tag = prev_tag
                    # if we found a best previous tag
                    if best_previous_tag is not None:
                         # set probability in our Viterbi matrix
                         V[j][curr_tag] = max_prob
                         # set our backtrack dict
                         new_dict[curr_tag] = dict[best_previous_tag] + [curr_tag]
            # update backtrack dicts from old to new
            dict = new_dict
        # init vars for finding the best final overall tag
        max_final_prob = float('-inf')
        best_final_tag = None

        # go thru the same process
        # iterate across all possible final tags
        for tag in V[seq_len - 1]:
            # get transition probability until an 'END' is reached
            trans_prob_end = trans_prob[tag].get('END', trans_prob[tag]['UNK'])
            # if zero...
            if trans_prob_end == 0:
                continue # skip
            # Calculate final probab including transition to 'END'
            final_prob = V[seq_len - 1][tag] + log(trans_prob_end)
            # if this is the best final probability
            if final_prob > max_final_prob:
                # update the vars like we've been doing
                max_final_prob = final_prob
                best_final_tag = tag
        # if we found a best final tag
        if best_final_tag is not None:
            # If there is a valid final tag, retrieve the best tag sequence from the backtrack dictionary
            best_tag_sequence = dict[best_final_tag]
        else:
            # If there is no valid final tag, find the most common tag
            # Initi a variable to store the most common tag
            most_common_tag = None
            # Initi a variable to store the max sum of emit prob
            max_sum = float('-inf')
            # Loop thru each tag in the emit prob
            for tag in emit_prob:
                # Calc the sum of emit prob for the current tag
                current_sum = sum(emit_prob[tag].values())
                # Check if the current sum is greater than the maximum sum found so far
                if current_sum > max_sum:
                    # Update the maximum sum
                    max_sum = current_sum
                    # Update the most common tag
                    most_common_tag = tag
            # Create a list of the most common tag repeated for the length of the sequence
            best_tag_sequence = [most_common_tag] * seq_len
        # final predicted sequence with START and END tokens
        predicted_sequence = [('START', 'START')] + [(curr_sentence[i], best_tag_sequence[i]) for i in range(seq_len)] + [('END', 'END')]
        # Append the predicted sequence to the list of predictions
        predicts.append(predicted_sequence)

    # Return predicts
    return predicts

    
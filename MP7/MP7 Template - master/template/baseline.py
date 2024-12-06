"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
        '''
        input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
                test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
        output: list of sentences, each sentence is a list of (word,tag) pairs.
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''

        # 1) we begin by processing the training data
        # first, we create a dict to store the frequency distribution of tags for each word
        word_tags_freq = {}
        # we also need a dict to keep track of the most common tag for unseen words
        most_common_tag = {}
        # processing the training data
        # from SP2023 MP08: for all seen word w, tag_w = argmax (# times tag t is matched to word w)
        # from SP2023 MP08: for all unseen word w, tag_w = argmax (# times tag t appears in the training set)
        for sentences in train: # loop thru list of sentences
                for word, tag in sentences: # tags and words in each sentence
                        if word not in word_tags_freq: # if the word is not found in our frequency distribution
                                word_tags_freq[word] = {} # create an empty element to mark it as its first instance
                        if tag not in word_tags_freq[word]: # if the tag is not found in our most common tag
                                word_tags_freq[word][tag] = 0 # initialize instance of the tag as well
                        word_tags_freq[word][tag] += 1 # increment word, tag count in our dict
                        if tag not in most_common_tag: # if not found in our dict
                                most_common_tag[tag] = 0 # mark it in our MCT dict
                        most_common_tag[tag] += 1 # increment tag, in our MCT dict
        # isolate the MCT from the most_common_tag dict
        mct = max(most_common_tag, key = most_common_tag.get)

        # 2) we tag the test data
        # first, we create an arr to store the tagged sentences --> as specified by the comments at the top
        tagged_sentences = []
        # loop thru testing set
        for sentences in test:
                # create an arr to store the individual tagged sentence
                tagged_sentence = []
                # loop thru each word
                for word in sentences: 
                        if word in word_tags_freq: # if word found in our freq distrib
                                # assign the MFT for the given word
                                mft = max(word_tags_freq[word], key = word_tags_freq[word].get)
                        else: # if not found in freq distrib
                                mft = mct # just assign the MCT to MFT
                        tagged_sentence.append((word, mft)) # append the final word + tag to the respective sentence
                tagged_sentences.append(tagged_sentence) # append the final sentence to the array of ALL sentences

        # return final result in specified format
        return tagged_sentences

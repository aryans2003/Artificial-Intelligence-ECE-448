# ECE-448-Coursework
MP (Machine Problems) from ECE 448: Artifical Intelligence @ UIUC FA'24

Brief Overview:

MP1:

Naive Bayes Classifier

This project contains a Python implementation of a Naive Bayes classifier for text classification tasks. The primary focus of this script is to train and predict text labels based on a dataset of reviews, using Laplace smoothing and a prior probability for classification. Code can be found in naive_bayes.py.

MP2:

Bigram Naive Bayes Classifier

This project features a Python implementation of a Bigram Naive Bayes classifier for text classification. This enhanced model incorporates both unigrams and bigrams to improve classification accuracy, utilizing Laplace smoothing for both unigrams and bigrams, and a mixture model to balance their contributions. Code can be found in bigram_naive_bayes.py

MP3:

Best First Search Algorithm

This project implements a Best First Search Algorithm designed to efficiently find the shortest path from a starting state to a goal state in a given problem space. The algorithm is encapsulated in the best_first_search function within search.py, where it utilizes a priority queue to explore neighboring states based on their heuristic values. The neighboring states and their properties are defined in the AbstractState class, which can be found in state.py. This class also includes methods for calculating the Manhattan distance heuristic and determining valid neighboring states for exploration, enabling the search algorithm to make informed decisions during its execution.

MP4:

Grid State Implementation

The Grid State Implementation project includes various classes designed to manage grid-based states for pathfinding scenarios. The core classes are SingleGoalGridState, MultiGoalGridState, and MultiAgentGridState, each of which extends functionality for different types of goal management within a grid. All code resides in state.py, where each class defines methods for obtaining neighboring states through the get_neighbors function. The SingleGoalGridState class wraps around maze neighbors to create new instances for reachable neighboring states and implements methods for checking if the goal is reached, computing the Manhattan distance as a heuristic, and defining equality and hash functions. The MultiGoalGridState class extends this by managing multiple goals, ensuring that states leading to reached goals are updated appropriately, while also utilizing the minimum spanning tree (MST) concept for heuristic calculation. The MultiAgentGridState class takes it a step further by managing multiple agents, checking for collisions and path crossing, while supporting both admissible and inadmissible heuristics for more efficient state exploration.

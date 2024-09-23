from utils import is_english_word, levenshteinDistance
from abc import ABC, abstractmethod
import copy # you may want to use copy when creating neighbors for EightPuzzle...

# NOTE: using this global index means that if we solve multiple 
#       searches consecutively the index doesn't reset to 0...
from itertools import count
global_index = count()

# TODO(III): You should read through this abstract class
#           your search implementation must work with this API,
#           namely your search will need to call is_goal() and get_neighbors()
class AbstractState(ABC):
    def __init__(self, state, goal, dist_from_start=0, use_heuristic=True):
        self.state = state
        self.goal = goal
        # we tiebreak based on the order that the state was created/found
        self.tiebreak_idx = next(global_index)
        # dist_from_start is classically called "g" when describing A*, i.e., f(state) = g(start, state) + h(state, goal)
        self.dist_from_start = dist_from_start
        self.use_heuristic = use_heuristic
        if use_heuristic:
            self.h = self.compute_heuristic()
        else:
            self.h = 0

    # To search a space we will iteratively call self.get_neighbors()
    # Return a list of AbstractState objects
    @abstractmethod
    def get_neighbors(self):
        pass
    
    # Return True if the state is the goal
    @abstractmethod
    def is_goal(self):
        pass
    
    # A* requires we compute a heuristic from each state
    # compute_heuristic should depend on self.state and self.goal
    # Return a float
    @abstractmethod
    def compute_heuristic(self):
        pass
    
    # The "less than" method ensures that states are comparable, meaning we can place them in a priority queue
    # You should compare states based on f = g + h = self.dist_from_start + self.h
    # Return True if self is less than other
    @abstractmethod
    def __lt__(self, other):
        # NOTE: if the two states (self and other) have the same f value, tiebreak using tiebreak_idx as below
        if self.tiebreak_idx < other.tiebreak_idx:
            return True

    # The "hash" method allow us to keep track of which states have been visited before in a dictionary
    # You should hash states based on self.state (and sometimes self.goal, if it can change)
    # Return a float
    @abstractmethod
    def __hash__(self):
        pass
    # __eq__ gets called during hashing collisions, without it Python checks object equality
    @abstractmethod
    def __eq__(self, other):
        pass
    
# WordLadder ------------------------------------------------------------------------------------------------

# TODO(III): we've provided you most of WordLadderState, read through our comments and code below.
#           The only thing you must do is fill in the WordLadderState.__lt__(self, other) method
class WordLadderState(AbstractState):
    def __init__(self, state, goal, dist_from_start, use_heuristic, cost_per_letter):
        '''
        state: string of length n
        goal: string of length n
        dist_from_start: integer
        use_heuristic: boolean
        '''
        # NOTE: AbstractState constructor does not take cost_per_letter
        super().__init__(state, goal, dist_from_start, use_heuristic)
        self.cost_per_letter = cost_per_letter
        
    # Each word can have many neighbors:
    #   Every letter in the word (self.state) can be replaced by every letter in the alphabet
    #   The resulting word must be a valid English word (i.e., in our dictionary)
    def get_neighbors(self):
        '''
        Return: a list of WordLadderState
        '''
        nbr_states = []
        for word_idx in range(len(self.state)):
            prefix = self.state[:word_idx]
            suffix = self.state[word_idx+1:]
            # 'a' = 97, 'z' = 97 + 25 = 122
            for c_idx in range(97, 97+26):
                c = chr(c_idx) # convert index to character
                # Replace the character at word_idx with c
                potential_nbr = prefix + c + suffix
                edge_cost = self.cost_per_letter[c]
                # If the resulting word is a valid english word, add it as a neighbor
                # NOTE: dist_from_start increases by edge_cost (this may not be 1!)
                if is_english_word(potential_nbr):
                    new_state = WordLadderState(
                        state=potential_nbr,
                        goal=self.goal, # stays the same!
                        dist_from_start=self.dist_from_start + edge_cost,
                        use_heuristic=self.use_heuristic, # stays the same!
                        cost_per_letter=self.cost_per_letter # stays the same!
                    )
                    nbr_states.append(new_state)
        return nbr_states

    # Checks if we reached the goal word with a simple string equality check
    def is_goal(self):
        return self.state == self.goal
    
    # Strings are hashable, directly hash self.state
    def __hash__(self):
        return hash(self.state)
    def __eq__(self, other):
        return self.state == other.state
    
    # The heuristic we use is the edit distance (Levenshtein) between our current word and the goal word
    def compute_heuristic(self):
        return levenshteinDistance(self.state, self.goal)
    
    # TODO(III): implement this method
    def __lt__(self, other):    
        # You should return True if the current state has a lower g + h value than "other"
        if(self.dist_from_start + self.h < other.dist_from_start + other.h): # g + h = .dist_from_start + .h
            return True
        # If they have the same value then you should use tiebreak_idx to decide which is smaller
        elif(self.dist_from_start + self.h == other.dist_from_start + other.h):
            return self.tiebreak_idx < other.tiebreak_idx # from 'less than' method
        return False # if all else fails, return false

    
    # str and repr just make output more readable when you print out states
    def __str__(self):
        return self.state
    def __repr__(self):
        return self.state

# EightPuzzle ------------------------------------------------------------------------------------------------

# TODO(IV): implement this method (also need it for the next homework)
# Manhattan distance between two points (a=(a1,a2), b=(b1,b2))
def manhattan(a, b):
    # Manhattan distance formula: |a1 - b1| + |a2 - b2|
    # if a = (a1, a2) then 'a' must be a tuple
    return ( abs(a[0] - b[0]) + abs(a[1] - b[1]) )

class EightPuzzleState(AbstractState):
    def __init__(self, state, goal, dist_from_start, use_heuristic, zero_loc):
        '''
        state: 3x3 array of integers 0-8
        goal: 3x3 goal array, default is np.arange(9).reshape(3,3).tolist()
        zero_loc: an additional helper argument indicating the 2d index of 0 in state, you do not have to use it
        '''
        # NOTE: AbstractState constructor does not take zero_loc
        super().__init__(state, goal, dist_from_start, use_heuristic)
        self.zero_loc = zero_loc
    
    # TODO(IV): implement this method
    def get_neighbors(self):
        '''
        Return: a list of EightPuzzleState
        '''
        nbr_states = []
        # NOTE: There are *up to 4* possible neighbors and the order you add them matters for tiebreaking
        #   Please add them in the following order: [below, left, above, right], where for example "below" 
        #   corresponds to moving the empty tile down (moving the tile below the empty tile up)

        # initializing function-scope variables here
        rows = 3 # 3x3 matrix
        cols = 3 # 3x3 matrix
        empty_row = -1 # row position for empty tile, set to out of bounds index
        empty_col = -1 # col position for empty tile, set to out of bounds index
        moved_row = 0 # row position for moved tile
        moved_col = 0 # col position for moved tile
        new_row = 0   # row position for new position
        new_col = 0   # col position for new position
        new_state = [] # arr to store new state of tile
        # possible moves in given order: below, left, above, right
        # (1, 0) --> row + 1 --> empty tile moves down
        # (0,-1) --> col - 1 --> empty tile moves left
        # (-1,0) --> row - 1 --> empty tile moves up
        # (0, 1) --> col + 1 --> empty tile moves right
        moves = [(1, 0), (0, -1), (-1, 0), (0, 1)] # stored as tuples in given arr
        

        # find empty tile
        for row in range(rows): # loop thru rows
            for col in range(cols): # loop thru cols
                if self.state[row][col] == 0: # if we found the empty tile
                    empty_row = row # store row
                    empty_col = col # store col
                    break # break out of inner loop
            if empty_row != -1:  # if empty tile found, exit outer loop as we can stop searching
                break # break out of outer loop

        # checking possible moves, calculating new position, error-checking, creating new puzzle state
        for moved_row, moved_col in moves: # loop thru possible moves: below, left, above, right (IN ORDER)
            # calculate new position with distance from empty position and moved position
            new_row = empty_row + moved_row
            new_col = empty_col + moved_col

            # ensure new position is within puzzle boundaries
            # both within 3x3 matrix
            if 0 <= new_row and new_row < rows and 0 <= new_col and new_col < cols:
                # if they are in a valid position inside the puzzle, we can create a deep copy
                new_state = copy.deepcopy(self.state) # use copy as mentioned at top

                # swap empty tile for neighboring tile
                new_state[empty_row][empty_col] = new_state[new_row][new_col] # swap new state's empty (row,col) with the new (row,col)
                new_state[new_row][new_col] = 0 # mark originally 'new' (row,col) as a now empty (0)

                # create a new instance of EightPuzzleState with updated vars:
                # we use our new state
                # self.goal
                # increment distance from start by 1 as we moved
                # self.use_heuristic
                # (new_row, new_col) coordinate position
                new_EightPuzzleState = EightPuzzleState(new_state, self.goal, (self.dist_from_start + 1), self.use_heuristic, (new_row, new_col))

                # append states to list of neighbors and continue onto next tuple
                nbr_states.append(new_EightPuzzleState)

        return nbr_states # return new EightPuzzleState

    # Checks if goal has been reached
    def is_goal(self):
        # In python "==" performs deep list equality checking, so this works as desired
        return self.state == self.goal
    
    # Can't hash a list, so first flatten the 2d array and then turn into tuple
    def __hash__(self):
        return hash(tuple([item for sublist in self.state for item in sublist]))
    def __eq__(self, other):
        return self.state == other.state
    
    # TODO(IV): implement this method
    def compute_heuristic(self):
        total = 0
        # NOTE: There is more than one possible heuristic, 
        #       please implement the Manhattan heuristic, as described in the MP instructions

        # init function-scope vars here
        rows = 3 # 3x3 grid
        cols = 3 # 3x3 grid
        curr_tile = 0 # temp var for current tile
        goal_list = {} # list for storing goal_coords in tuples
        goal_coords = 0 # var to store values of goal coordinates

        # computing goal positions
        for goal_row in range(rows): # loop thru rows
            for goal_col in range(cols): # loop thru cols
                if self.goal[goal_row][goal_col] != 0: # if tile in the goal state is a valid tile
                    goal_list[self.goal[goal_row][goal_col]] = (goal_row, goal_col) # add goal coordinates to list as a tuple at correct index

        # computing total by verifying all current tiles
        for row in range(rows): # loop thru rows
            for col in range(cols): # loop thru cols
                curr_tile = self.state[row][col] # get self state of current tile
                if curr_tile != 0: # skip empty tile 
                    goal_coords = goal_list[curr_tile] # set goal cooridnates to the goal position of our current tile
                    # use helper manhattan funct and add goal_coords to total
                    total += manhattan((row, col), goal_coords)
        return total
    
    # TODO(IV): implement this method
    # Hint: it should be identical to what you wrote in WordLadder.__lt__(self, other)
    def __lt__(self, other):
        # You should return True if the current state has a lower g + h value than "other"
        if(self.dist_from_start + self.h < other.dist_from_start + other.h): # g + h = .dist_from_start + .h
            return True
        # If they have the same value then you should use tiebreak_idx to decide which is smaller
        elif(self.dist_from_start + self.h == other.dist_from_start + other.h):
            return self.tiebreak_idx < other.tiebreak_idx # from 'less than' method
        return False # if all else fails, return false
    
    # str and repr just make output more readable when you print out states
    def __str__(self):
        return self.state
    def __repr__(self):
        return "\n---\n"+"\n".join([" ".join([str(r) for r in c]) for c in self.state])
    

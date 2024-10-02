from abc import ABC, abstractmethod
from itertools import count, product
import numpy as np

from utils import compute_mst_cost

# NOTE: using this global index (for tiebreaking) means that if we solve multiple 
#       searches consecutively the index doesn't reset to 0... this is fine
global_index = count()

# Manhattan distance between two (x,y) points
def manhattan(a, b):
    # TODO(III): you should copy your code from MP3 here
    # Manhattan distance formula: |a1 - b1| + |a2 - b2|
    # if a = (a1, a2) then 'a' must be a tuple
    return ( abs(a[0] - b[0]) + abs(a[1] - b[1]) )

class AbstractState(ABC):
    def __init__(self, state, goal, dist_from_start=0, use_heuristic=True):
        self.state = state
        self.goal = goal
        # we tiebreak based on the order that the state was created/found
        self.tiebreak_idx = next(global_index)
        # dist_from_start is classically called "g" when describing A*, i.e., f = g + h
        self.dist_from_start = dist_from_start
        self.use_heuristic = use_heuristic
        if use_heuristic:
            self.h = self.compute_heuristic()
        else:
            self.h = 0

    # To search a space we will iteratively call self.get_neighbors()
    # Return a list of State objects
    @abstractmethod
    def get_neighbors(self):
        pass
    
    # Return True if the state is the goal
    @abstractmethod
    def is_goal(self):
        pass
    
    # A* requires we compute a heuristic from eahc state
    # compute_heuristic should depend on self.state and self.goal
    # Return a float
    @abstractmethod
    def compute_heuristic(self):
        pass
    
    # Return True if self is less than other
    # This method allows the heap to sort States according to f = g + h value
    def __lt__(self, other):
        # TODO(III): you should copy your code from MP3 here
        if(self.dist_from_start + self.h < other.dist_from_start + other.h): # g + h = .dist_from_start + .h
            return True
        # If they have the same value then you should use tiebreak_idx to decide which is smaller
        elif(self.dist_from_start + self.h == other.dist_from_start + other.h):
            return self.tiebreak_idx < other.tiebreak_idx # from 'less than' method
        return False # if all else fails, return false

    # __hash__ method allow us to keep track of which 
    #   states have been visited before in a dictionary
    # You should hash states based on self.state (and sometimes self.goal, if it can change)
    # Return a float
    @abstractmethod
    def __hash__(self):
        pass
    # __eq__ gets called during hashing collisions, without it Python checks object equality
    @abstractmethod
    def __eq__(self, other):
        pass
    
class SingleGoalGridState(AbstractState):
    # state: a length 2 tuple indicating the current location in the grid, e.g., (x, y)
    # goal: a length 2 tuple indicating the goal location, e.g., (x, y)
    # maze_neighbors: function for finding neighbors on the grid (deals with checking collision with walls...)
    def __init__(self, state, goal, dist_from_start, use_heuristic, maze_neighbors):
        self.maze_neighbors = maze_neighbors
        super().__init__(state, goal, dist_from_start, use_heuristic)
        
    # This is basically just a wrapper for self.maze_neighbors
    def get_neighbors(self):
        nbr_states = []
        # neighboring_locs is a tuple of tuples of neighboring locations, e.g., ((x1, y1), (x2, y2), ...)
        # feel free to look into maze.py for more details
        neighboring_locs = self.maze_neighbors(*self.state)
        # TODO(III): fill this in
        # The distance from the start to a neighbor is always 1 more than the distance to the current state
        # -------------------------------

        # in MP3, for EightPuzzleState get_neighbors, we returned a list of instances of EightPuzzleState
        # so for this, we want to return a list of instances of SingleGoalGridState

        # loop through each location among the neighboring locations
        for location in neighboring_locs:
                # create a new instance of SingleGoalGridState with updated vars:
                # we use our new location
                # self.goal
                # increment distance from start by 1 as we moved
                # self.use_heuristic
                # self.maze_neighbors coordinate position
                new_SingleGoalGridState = SingleGoalGridState(location, self.goal, (self.dist_from_start + 1), self.use_heuristic, self.maze_neighbors)

                # append states to list of neighbors and continue onto next tuple
                nbr_states.append(new_SingleGoalGridState)

        # -------------------------------
        return nbr_states

    # TODO(III): fill in the is_goal, compute_heuristic, __hash__, and __eq__ methods
    # Your heuristic should be the manhattan distance between the state and the goal

    # Checks if goal has been reached
    def is_goal(self):
        # In python "==" performs deep list equality checking, so this works as desired
        # curr loc is self.state, curr goal is self.goal, just check if they equal
        return self.state == self.goal
    
    def compute_heuristic(self):
        # your heuristic should be the manhattan distance, use manhattan funct
        # between state and goal
        return manhattan(self.state, self.goal)
    
    # Can't hash a list, so first flatten the 2d array and then turn into tuple
    def __hash__(self):
        # convert self.state to hash (tuple) and compute hash val
        return hash(self.state)
    
    # same as from MP3
    def __eq__(self, other):
        # true if 'state' of curr == 'state' of other
        return self.state == other.state
    
    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state)
    def __repr__(self):
        return str(self.state)

class MultiGoalGridState(AbstractState):
    # state: a length 2 tuple indicating the current location in the grid, e.g., (x, y)
    # goal: a tuple of length 2 tuples of locations in the grid that have not yet been reached
    #       e.g., ((x1, y1), (x2, y2), ...)
    # maze_neighbors: function for finding neighbors on the grid (deals with checking collision with walls...)
    # mst_cache: reference to a dictionary which caches a set of goal locations to their MST value
    def __init__(self, state, goal, dist_from_start, use_heuristic, maze_neighbors, mst_cache):
        self.maze_neighbors = maze_neighbors
        self.mst_cache = mst_cache
        super().__init__(state, goal, dist_from_start, use_heuristic)
        
    # We get the list of neighbors from maze_neighbors
    # Then we need to check if we've reached one of the goals, and if so remove it
    def get_neighbors(self):
        nbr_states = []
        # neighboring_locs is a tuple of tuples of neighboring locations, e.g., ((x1, y1), (x2, y2), ...)
        # feel free to look into maze.py for more details
        neighboring_locs = self.maze_neighbors(*self.state)
        # TODO(IV): fill this in
        # -------------------------------

        for location in neighboring_locs: # loop thru locs in neighboring_locs
            new_goals = [] # create list for new_goals to be stored in
            for goal in self.goal: # check each goal in self.goal
                if goal != location: # make sure goal does not equal location
                    new_goals.append(goal) # append new goals to list

            new_goals = tuple(new_goals) # convert list into tuple
                
            # create a new instance of MultiGoalGridState with updated vars:
            # we use our new location
            # new_goals
            # increment distance from start by 1 as we moved
            # self.use_heuristic
            # self.maze_neighbors coordinate position
            # self.mst_cache is needed for MultiGoalGridState
            new_MultiGoalGridState = MultiGoalGridState(location, new_goals, (self.dist_from_start + 1), self.use_heuristic, self.maze_neighbors, self.mst_cache)

            # append states to list of neighbors and continue onto next tuple
            nbr_states.append(new_MultiGoalGridState)

        # -------------------------------
        return nbr_states

    # TODO(IV): fill in the is_goal, compute_heuristic, __hash__, and __eq__ methods
    # Your heuristic should be the cost of the minimum spanning tree of the remaining goals 
    #   plus the manhattan distance to the closest goal
    #   (you should use the mst_cache to store the MST values)
    # Think very carefully about your eq and hash methods, is it enough to just hash the state?

    def is_goal(self):
        # is_goal is true if there are no more goals left to reach
        return len(self.goal) == 0
    
    def __hash__(self):
        # we want to hash both the current and remaining goals, not just the state as mentioned
        return hash((self.state, self.goal)) # check concatenation method
    
    def compute_heuristic(self):
        # if no goals, return 0
        if not self.goal:
            return 0

        # use some arbritary high max value, temp var for closest dist
        closest_goal_dist = 999
        # loop thru goals
        for goal in self.goal:
            # calc curr goal distance
            curr_goal_dist = manhattan(self.state, goal)
            # check if less than curr closest goal dist
            if curr_goal_dist < closest_goal_dist:
                # if so, update closest goal dist
                closest_goal_dist = curr_goal_dist

        # create an MST cache key for the cost
        # store it in a sorted tuple
        mst_cache_key = tuple(sorted(self.goal))
        # retrieve MST cost from the cache key
        mst_cost = self.mst_cache.get(mst_cache_key)
        # if no cost was in the cache
        if mst_cost is None:
            # compute the cost using the compute_mst_cost in the utils.py
            # based on goal and manhattan dist
            mst_cost = compute_mst_cost(self.goal, manhattan)
            # update mst_cache val, needed for MultiGoalGridState
            self.mst_cache[mst_cache_key] = mst_cost
        # calculate heuristic based on closest distance and MST cost
        heuristic = closest_goal_dist + mst_cost
        # return
        return heuristic
    
    def __eq__(self, other):
        # same as always
        return self.state == other.state
    
    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state) + ", goals=" + str(self.goal)
    def __repr__(self):
        return str(self.state) + ", goals=" + str(self.goal)
    
class MultiAgentGridState(AbstractState):
    # state: a tuple of agent locations
    # goal: a tuple of goal locations for each agent
    # maze_neighbors: function for finding neighbors on the grid
    #   NOTE: it deals with checking collision with walls... but not with other agents
    def __init__(self, state, goal, dist_from_start, use_heuristic, maze_neighbors, h_type="admissible"):
        self.maze_neighbors = maze_neighbors
        self.h_type = h_type
        super().__init__(state, goal, dist_from_start, use_heuristic)
        
    # We get the list of neighbors for each agent from maze_neighbors
    # Then we need to check inter agent collision and inter agent edge collision (crossing paths)
    def get_neighbors(self):
        nbr_states = []
        neighboring_locs = [self.maze_neighbors(*s) for s in self.state]
        for nbr_locs in product(*neighboring_locs):
            # TODO(V): fill this in
            # You will need to check whether two agents collide or cross paths
            #   - Agents collide if they move to the same location 
            #       - i.e., if there are any duplicate locations in nbr_locs
            #   - Agents cross paths if they swap locations
            #       - i.e., if there is some agent whose current location (in self.state) 
            #       is the same as the next location of another agent (in nbr_locs) *and vice versa*
            # Before writing code you might want to understand what the above lines of code do...
            # -------------------------------
            
            # check whether two agents collide

            # check if they move to the same location
            if len(set(nbr_locs)) != len(nbr_locs): # check for duplicate locations in nbr_locs
                continue # skip as we found collision

            # check if they swap locations / cross paths
            if any(
                    # for each agent i --> python any loop
                    # check if any pair of agents swap positions
                    self.state[i] == nbr_locs[j] and self.state[j] == nbr_locs[i] 
                    # iterate over each agent 'i' in the state
                    for i in range(len(self.state)) 
                    # iterate over each agent 'j' that is after curr agent 'i'
                    for j in range(i + 1, len(self.state))):
                # if we find a crossed path, we can skip the curr neighbor
                continue

            # create a new instance of MultiAgentGridState with updated vars:
            # nbr_locs as location
            # self.goal
            # increment distance from start by 1 as we moved
            # self.use_heuristic
            # self.maze_neighbors coordinate position
            # use self.h_type to determine which heuristic to use
            new_MultiAgentGridState = MultiAgentGridState(nbr_locs, self.goal, (self.dist_from_start + 1), self.use_heuristic, self.maze_neighbors, self.h_type)

            # append states to list of neighbors and continue onto next tuple
            nbr_states.append(new_MultiAgentGridState) 
            # -------------------------------            
        return nbr_states
    
    def compute_heuristic(self):
        if self.h_type == "admissible":
            return self.compute_heuristic_admissible()
        elif self.h_type == "inadmissible":
            return self.compute_heuristic_inadmissible()
        else:
            raise ValueError("Invalid heuristic type")

    # TODO(V): fill in the compute_heuristic_admissible and compute_heuristic_inadmissible methods
    #   as well as the is_goal, __hash__, and __eq__ methods
    # As implied, in compute_heuristic_admissible you should implement an admissible heuristic
    #   and in compute_heuristic_inadmissible you should implement an inadmissible heuristic 
    #   that explores fewer states but may find a suboptimal path
    # Your heuristics should be at least as good as ours on the autograder 
    #   (with respect to number of states explored and path length)

    def compute_heuristic_admissible(self):
        max_dist = 0 # temp var to hold max distance
        # loop thru states
        for i in range(len(self.state)):
            # compute distance as manhattan distance from state to goal
            dist = manhattan(self.state[i], self.goal[i])
            # check if computed distance > our current max
            if dist > max_dist:
                # if so, update max
                max_dist = dist
        # return max after all states checked
        return max_dist
    
    def compute_heuristic_inadmissible(self):
        total_dist = 0  # initialize total distance
        # loop thru states
        for i in range(len(self.state)):
            # compute distance as manhattan distance from state to goal
            dist = manhattan(self.state[i], self.goal[i])
            # add to curr dist to total dist
            total_dist += dist
    
        return total_dist
    
    def is_goal(self):
        # in multi agent, a state is a goal if all agents reached their goals
        for agent_pos, goal_pos in zip(self.state, self.goal): # loop thru agents and goals
            if agent_pos != goal_pos: # check if agents reached goals
                return False # if not, return false
        return True # after looping, none found, return true
    
    def __hash__(self):
        # same as multi goal grid state
        return hash((self.state, self.goal))

    def __eq__(self, other):
        # same as multi goal grid state, maybe add a check for goal??
        return self.state == other.state
    
    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state) + ", goals=" + str(self.goal)
    def __repr__(self):
        return str(self.state) + ", goals=" + str(self.goal)
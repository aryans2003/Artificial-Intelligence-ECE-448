import heapq

def best_first_search(starting_state):
    # TODO(III): You should copy your code from MP3 here
    '''
    Implementation of best first search algorithm

    Input:
        starting_state: an AbstractState object

    Return:
        A path consisting of a list of AbstractState states
        The first state should be starting_state
        The last state should have state.is_goal() == True
    '''
    # we will use this visited_states dictionary to serve multiple purposes
    # - visited_states[state] = (parent_state, distance_of_state_from_start)
    #   - keep track of which states have been visited by the search algorithm
    #   - keep track of the parent of each state, so we can call backtrack(visited_states, goal_state) and obtain the path
    #   - keep track of the distance of each state from start node
    #       - if we find a shorter path to the same state we can update with the new state 
    # NOTE: we can hash states because the __hash__/__eq__ method of AbstractState is implemented
    visited_states = {starting_state: (None, 0)} # use a dict to keep track of visited states -- DONE

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = [] # use a priority queue of states on the 'frontier' -- DONE
    heapq.heappush(frontier, starting_state)
    
    # TODO(III): implement the rest of the best first search algorithm
    # HINTS:
    #   - add new states to the frontier by calling state.get_neighbors()
    #   - check whether you've finished the search by calling state.is_goal()
    #       - then call backtrack(visited_states, state)...
    # Your code here ---------------

    # iteratively search through the neighbors of each state until you find the shortest path to the goal
    # using pseudocode from lecture 3

    # while loop through the heap's contents
    while frontier:
        u = heapq.heappop(frontier) # pick a state from the frontier, since BFS --> order explored (FIFO)
        if u.is_goal(): # check if it is_goal
            return backtrack(visited_states, u) # if so, done call backtrack... as specified
        for neighbor in u.get_neighbors(): # explore neighbors
            if neighbor in visited_states: # if neighbor been visited
                # we are checking state paths based on distance from start + cost heuristic (both given from AbstractClass)
                # f = g + h = .dist_from_start + .h
                if visited_states[neighbor][1] > neighbor.dist_from_start + neighbor.h: # found a shorter path
                    visited_states[neighbor] = (u, neighbor.dist_from_start + neighbor.h) # update visited state
                    heapq.heappush(frontier, neighbor) # update frontier
            elif neighbor not in visited_states: # if neighbor has not been visited
                visited_states[neighbor] = (u, neighbor.dist_from_start + neighbor.h) # update visited state
                heapq.heappush(frontier, neighbor) # update frontier
        
    # ------------------------------
    
    # if you do not find the goal return an empty list
    return []

def backtrack(visited_states, goal_state):
    # TODO(III): You should copy your code from MP3 here
    path = []
    # Your code here ---------------
    curr = goal_state # set curr to goal_state for backtracking
    while curr is not None: # continue to backtrack until goal is empty
        path.append(curr) # add curr goal state to path list
        curr = visited_states[curr][0] # update goal to parent state ([curr][0])
    path.reverse() # reverse backwards list to get right order (start-->goal)
    # ------------------------------
    return path
# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
import heapq


# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze


def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


# TODO: VI
def astar(maze):
    # get the starting state from the maze
    starting_state = maze.get_start()

    # we will use this visited_states dictionary to serve multiple purposes
    # - visited_states[state] = (parent_state, distance_of_state_from_start)
    #   - keep track of which states have been visited by the search algorithm
    #   - keep track of the parent of each state, so we can call backtrack(visited_states, goal_state) and obtain the path
    #   - keep track of the distance of each state from start node
    #       - if we find a shorter path to the same state we can update with the new state 
    # NOTE: we can hash states because the __hash__/__eq__ method of AbstractState is implemented
    visited_states = {starting_state: (None, 0)} # use a dict to keep track of visited states + costs now for A*

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = [] # use a priority queue of states on the 'frontier' -- DONE
    heapq.heappush(frontier, (starting_state.h, starting_state)) # now we also push the heuristic

    # modifying code from my MP 3 / 4

    # while loop through the heap's contents
    while frontier:
        current_f, u = heapq.heappop(frontier) # pop the state with the lowest f = g + h cost
        if maze.is_objective((u.x, u.y)): # if current state is a goal state
            return backtrack(visited_states, u) # if so, done call backtrack... as specified
        for neighbor in maze.get_neighbors(u.x, u.y, u.shape_idx):  # Get neighbors from maze class
            g_cost = visited_states[u][1] + 1  # Assuming uniform cost, update if needed based on your move cost
            if neighbor in visited_states:  # If the neighbor has been visited
                # Compare the existing cost with the new cost
                if visited_states[neighbor][1] > g_cost:  # Found a shorter path
                    visited_states[neighbor] = (u, g_cost)  # Update visited state
                    heapq.heappush(frontier, (g_cost + neighbor.h, neighbor))  # Push onto frontier with new f value
            else:  # If the neighbor has not been visited
                visited_states[neighbor] = (u, g_cost)  # Record the state
                heapq.heappush(frontier, (g_cost + neighbor.h, neighbor))  # Push onto frontier with f value
        
    # ------------------------------
    
    # if you do not find a path, return None
    return None


# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
# TODO: VI
def backtrack(visited_states, current_state):
    path = []
    # Your code here ---------------
    curr = current_state # start from curr state
    while curr is not None: # continue to backtrack until we reach starting state
        path.append(curr) # add curr state to path
        curr = visited_states[curr] # move to parent state
    path.reverse() # reverse backwards list to get right order (start-->curr)
    return path

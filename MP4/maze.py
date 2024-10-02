# maze.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Amnon Attali (aattali2@illinois.edu) on 09/15/2024, 

import numpy as np

class Maze:
    def __init__(self, file_path, allow_waiting = False):
        """
        Creates a maze instance given a `file_path` to a text file containing the ASCII representation of a maze.
        Key:
            - Walls are represented by %
            - Open paths by spaces
            - Starts by capital letters (at most one of each letter) 
            - Goals (waypoints) by lowercase letters matching one of the starts
        
        If `allow_waiting` is True, the agent can stay in place (not move) as a valid action.
        """
        self.file_path = file_path
        self.allow_waiting = allow_waiting
        with open(file_path) as file:
            lines = tuple(line.strip() for line in file.readlines() if line)
        
        wall_char = '%'
        free_char = ' '
        height = len(lines)
        width = len(lines[0])
        
        # check that we have a rectangular grid
        if any(len(line) != width for line in lines):
            raise ValueError(f'(maze {file_path}): all maze rows must be the same length')
        
        # read the maze from the file into a numpy array and store starts and goals
        self.grid = np.zeros((height, width))
        self.starts = {}
        self.goals = {}
        for i in range(height):
            for j in range(width):
                cur_char = lines[i][j]
                if cur_char == wall_char:
                    self.grid[i, j] = 1
                elif cur_char.isupper():
                    if cur_char.lower() in self.starts:
                        raise ValueError(f'(maze {file_path}): starts must be unique')
                    self.starts[cur_char.lower()] = (i, j)
                elif cur_char.islower():
                    if cur_char not in self.goals:
                        self.goals[cur_char] = ()
                    self.goals[cur_char] += ((i, j),)
                elif cur_char != free_char:
                    raise ValueError(f'(maze {file_path}): invalid character {cur_char}')        
                
        # check that every start has a corresponding goal
        for start in self.starts:
            if start not in self.goals:
                raise ValueError(f'(maze {file_path}): start {start} has no corresponding goal')

        # check that border contains walls
        if np.any(self.grid[0, :]==0) or\
            np.any(self.grid[-1, :]==0) or\
                np.any(self.grid[:, 0]==0) or\
                    np.any(self.grid[:, -1]==0):
            raise ValueError(f'(maze {file_path}): border of maze must be walls')

        # This is a helper to track number of states (i,j) for which we call self.is_free(i, j)
        # This is a proxy for "number of collision detections" in robotics
        self.num_states_validated = 0
    
    def in_bounds(self, i, j):
        """Check if cell (i,j) is in bounds"""
        return 0 <= i < self.grid.shape[0] and 0 <= j < self.grid.shape[1]
    
    def indices(self):
        """Returns generator of all indices in maze"""
        return zip(*np.indices(self.grid.shape).reshape(2,-1))
    
    def is_free(self, i, j):
        """Check if cell (i,j) is free - not a wall"""
        self.num_states_validated += 1
        return self.in_bounds(i, j) and self.grid[i, j] != 1

    def neighbors(self, i, j):
        """Returns tuple of neighboring cells that can be moved to from the given row,col"""
        possible_moves = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
        if self.allow_waiting:
            possible_moves.append((i, j))
        return tuple(x for x in possible_moves if self.is_free(*x))

    def valid_path(self, path):
        # validate type and shape 
        if len(path) == 0:
            print(f'Invalid path: path must contain at least one element')
            return False
        if not all(len(vertex) == 2 for vertex in path):
            print(f'Invalid path: each path element must be a two-element sequence')
            return False
        
        # normalize path in case student used an element type that is not `tuple` 
        path = tuple(map(tuple, path))

        # check if path is contiguous
        for i, (a, b) in enumerate(zip(path, path[1:])):
            d = sum(abs(b_ - a_) for a_, b_ in zip(a, b)) 
            if d > 1:
                print(f'Invalid path: path vertex {i} {a} is too far from consecutive path vertex {i + 1} {b}')
                return False
            if d == 0 and not self.allow_waiting:
                print(f'Invalid path: path vertex {i} {a} is the same as consecutive path vertex {i + 1} {b}, and waiting is not allowed')
                return False

        # check if path is navigable 
        for i, x in enumerate(path):
            if not self.is_free( * x ):
                print(f'Invalid path: path vertex {i} {x} is not a navigable maze cell')
                return False
        
        # the path must start at a start location
        if path[0] not in self.starts.values():
            print(f'Invalid path: first path vertex {path[0]} must be a start location')
            return False
        
        # get the goal associated with the path start
        path_goals = None
        for c, start in self.starts.items():
            if start == path[0]:
                path_goals = self.goals[c]                
                break
        
        # check if the path ends at a goal 
        if path[-1] not in path_goals:
            print(f'Invalid path: last path vertex {path[-1]} must be a goal')
            return False
        
        # check for unnecessary path segments (looping back to a previous location without visiting a waypoint)
        if not self.allow_waiting:
            indices = {}
            for i, x in enumerate(path):
                if x in indices:
                    if all(x not in path_goals for x in path[indices[x] : i]):
                        print(f'Bad path: path segment [{indices[x]} : {i}] contains no waypoints but loops back to a previous location')
                        return False
                indices[x] = i 
        
        # check if path contains all waypoints 
        for goal in path_goals:
            if goal not in path:
                print(f'Bad path: path must contain all waypoints')
                return False
        
        return True


# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021, 
# Inspired by previous work by Michael Abir (abir2@illinois.edu) and Rahul Kunji (rahulsk2@illinois.edu)

import argparse
import pygame

class gradient:
    def __init__(self, start, end):
        # rgb colors
        self.start  = start 
        self.end    = end 
    
    def __getitem__(self, fraction):
        t = fraction[0] / max(1, fraction[1] - 1) # prevent division by zero
        return tuple(max(0, min(start * (1 - t) + end * t, 255)) 
            for start, end in zip(self.start, self.end))

class agent:
    def __init__(self, position, maze : Maze):
        self.position = position 
        self.maze = maze 

    def move(self, move):
        position = tuple(i + move for i, move in zip(self.position, move))
        if self.maze.is_free( * position ):
            previous = self.position
            self.position = position 
            return previous,
        else: 
            return ()
            
class Application:
    def __init__(self, human = True, scale = 20, fps = 30, alt_color = False):
        self.running    = True
        self.scale      = scale
        self.fps        = fps
        
        self.human      = human 
        # accessibility for colorblind students 
        if alt_color:
            self.gradient = gradient((64, 224, 208), (139, 0, 139))
        else:
            self.gradient = gradient((255, 0, 0), (0, 255, 0))

    def run(self, maze : Maze, path=[], save=None):
        self.maze = maze
        self.window = tuple(x * self.scale for x in reversed(self.maze.grid.shape))
        
        if self.human:
            if len(self.maze.starts) != 1:
                raise ValueError("Human player can only play mazes with exactly one start")
            self.agent = agent(list(self.maze.starts.values())[0], self.maze)
            path = []
            states_explored = 0
        else:
            states_explored = self.maze.num_states_validated

        pygame.init()
        self.font = pygame.font.SysFont('Arial', 16)
        self.surface = pygame.display.set_mode(self.window, pygame.HWSURFACE)
        self.surface.fill((255, 255, 255))
        pygame.display.flip()
        pygame.display.set_caption(maze.file_path)

        if self.human:
            self.draw_player()
        else:
            if len(self.maze.starts) != 1:
                print(f"Results: path length: {len(path[0])}, states explored: {states_explored}")
                num_agents = len(path)
                delta = 0.6 / num_agents
                offset = 0.2 + delta / 2
                for agent_path in path:
                    # self.draw_path(agent_path)
                    self.draw_path_lines(agent_path, offset=offset)
                    offset += delta
            else:
                print(f"Results: path length: {len(path)}, states explored: {states_explored}")
                self.draw_path(path)

        self.draw_maze()
        self.draw_start()
        self.draw_waypoints()

        pygame.display.flip()
        
        if type(save) is str:
            pygame.image.save(self.surface, save)
            self.running = False
        
        clock = pygame.time.Clock()
        
        while self.running:
            pygame.event.pump()
            clock.tick(self.fps)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise SystemExit
                elif event.type == pygame.KEYDOWN and self.human:
                    try:
                        move = {
                            pygame.K_RIGHT  : ( 0,  1),
                            pygame.K_LEFT   : ( 0, -1),
                            pygame.K_UP     : (-1,  0),
                            pygame.K_DOWN   : ( 1,  0),
                        }[event.key] 
                        path.extend(self.agent.move(move))
                    except KeyError: 
                        pass
                
                    self.loop(path + [self.agent.position])

    # The game loop is where everything is drawn to the context. Only called when a human is playing
    def loop(self, path):
        self.draw_path(path)
        self.draw_waypoints()
        self.draw_player()
        self.draw_start()
        pygame.display.flip()

    # Draws the path (given as a list of (row, col) tuples) to the display context
    def draw_path(self, path):
        for i, x in enumerate(path):
            self.draw_square(*x, self.gradient[i, len(path)])
    
    def draw_path_lines(self, path, offset = 0.5):
        for i in range(len(path)):
            if i > 0:
                pygame.draw.line(
                    self.surface, self.gradient[i, len(path)], 
                    tuple(int((x + offset) * self.scale) for x in reversed(path[i - 1])), 
                    tuple(int((x + offset) * self.scale) for x in reversed(path[i])), 
                    width=2)

    # Draws the full maze to the display context
    def draw_maze(self):
        for x in self.maze.indices():
            if not self.maze.is_free(*x): 
                self.draw_square(*x)
        # draw grid
        for i in range(self.maze.grid.shape[0] + 1):
            pygame.draw.line(self.surface, (0, 0, 0),
                            (0, i * self.scale), (self.window[0], i * self.scale), width=1)
        for i in range(self.maze.grid.shape[1] + 1):
            pygame.draw.line(self.surface, (0, 0, 0),
                            (i * self.scale, 0), (i * self.scale, self.window[1]), width=1)
    
    def draw_square(self, i, j, color = (0, 0, 0)):
        pygame.draw.rect(self.surface, color, tuple(i * self.scale for i in (j, i, 1, 1)), 0)
    
    def draw_circle(self, i, j, color = (0, 0, 0), radius = None):
        if radius is None:
            radius = self.scale / 4
        pygame.draw.circle(self.surface, color, tuple(int((i + 0.5) * self.scale) for i in (j, i)), int(radius))

    # Draws the player to the display context, and draws the path moved (only called if there is a human player)
    def draw_player(self):
        self.draw_circle( * self.agent.position , (0, 0, 255))

    # Draws the waypoints to the display context
    def draw_waypoints(self):
        for goal_char, goal_positions in self.maze.goals.items():
            for x in goal_positions:
                self.draw_circle(*x, color=(0, 255, 0))
                self.surface.blit(
                    self.font.render(goal_char, True, (0, 0, 0)), 
                    (int((x[1]+0.4) * self.scale), int((x[0]) * self.scale)))
        
    # Draws start location of path
    def draw_start(self):
        for start_char, (i,j) in self.maze.starts.items():
            # self.draw_circle(i, j, (0, 0, 255), self.scale / 2)
            pygame.draw.rect(self.surface, (255, 0, 0), tuple(int(i * self.scale) for i in (j + 0.25, i + 0.25, 0.5, 0.5)), 0)
            self.surface.blit(
                self.font.render(start_char.upper(), True, (0, 0, 0)), 
                (int((j+0.4) * self.scale), int((i) * self.scale)))
        

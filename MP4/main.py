from state import SingleGoalGridState, MultiGoalGridState, MultiAgentGridState
from search import best_first_search
from maze import Application, Maze

import time
import argparse

def main(args):
    if "Grid" in args.problem_type:
        filename = args.maze_file
        print(f"Doing Maze search for file {filename}")
        maze = Maze(filename, allow_waiting=args.allow_waiting)
        # maze.starts is a dictionary mapping characters to locations
        # maze.goals is a dictionary mapping characters to a tuple of locations
        tasks = [(maze.starts[c], maze.goals[c]) for c in maze.starts]
        starts, goals = zip(*tasks)
        path = []
        if not args.human:
            start_time = time.time()  
            if args.problem_type == "GridSingle":
                # single goal search
                start = starts[0] # only one start
                goal = goals[0][0] # only one goal for that one start
                starting_state = SingleGoalGridState(
                    start, goal, 
                    dist_from_start = 0, use_heuristic = args.use_heuristic, 
                    maze_neighbors = maze.neighbors)
            elif args.problem_type == "GridMultiGoal":
                # multi goal search
                start = starts[0] # only one start
                goal = goals[0] # only need goals for the one start
                starting_state = MultiGoalGridState(
                    start, goal, 
                    dist_from_start = 0, use_heuristic = args.use_heuristic,
                    maze_neighbors = maze.neighbors, mst_cache = {})
            elif args.problem_type == "GridMultiAgent":
                # multi agent search
                start = starts
                goal = tuple(g[0] for g in goals) # NOTE: we assume one goal per agent
                starting_state = MultiAgentGridState(
                    start, goal, 
                    dist_from_start = 0, use_heuristic = args.use_heuristic,
                    maze_neighbors = maze.neighbors, h_type = args.heuristic_type)
            # regardless of problem type, we can use the same search function
            path = best_first_search(starting_state)
            end_time = time.time()

            print("\tStart: ", start)
            print("\tGoal: ", goal)
            print("\tPath length: ", len(path))
            print("\tStates explored: ", maze.num_states_validated)
            print("\tTime:", end_time-start_time)
        
        if args.show_maze_vis or args.human:
            if args.problem_type == "GridSingle" or args.problem_type == "GridMultiGoal":
                vis_path = [s.state for s in path]
            elif args.problem_type == "GridMultiAgent":
                # need to transpose the path to visualize it
                vis_path = [[] for _ in range(len(starts))]
                for s in path:
                    for i, loc in enumerate(s.state):
                        vis_path[i].append(loc)
            application = Application(args.human, args.scale, args.fps, args.altcolor)
            application.run(maze, vis_path, args.save_maze)
    else:
        print("Problem type must be one of [GridSingle, GridMultiGoal, GridMultiAgent]")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP4 Search')

    parser.add_argument('--problem_type',dest="problem_type", type=str, default="GridSingle",
        help='Which search problem (i.e., AbstractState) to solve: [GridSingle, GridMultiGoal, GridMultiAgent]')
    parser.add_argument('--maze_file', type=str, default="data/grid_single/tiny",
                        help = 'path to maze file')
    parser.add_argument('--heuristic_type', type=str, default="admissible",
                        help = 'Which type of heuristic to use in multi agent search: [admissible, inadmissible]')
    parser.add_argument('--show_maze_vis', action = 'store_true',
                        help = 'show maze visualization')
    parser.add_argument('--allow_waiting', action = 'store_true',
                        help = 'adds a waiting action to the agent')
    parser.add_argument('--human', action = 'store_true',
                        help = 'run in human-playable mode')
    parser.add_argument('--use_heuristic', action = 'store_true',
                        help = 'use heuristic h in best_first_search')
    
    # Visualization ARGS: you do not need to change these
    parser.add_argument('--scale', dest = 'scale', type = int, default = 20,
                        help = 'display scale')
    parser.add_argument('--fps', dest = 'fps', type = int, default = 30,
                        help = 'display framerate')
    parser.add_argument('--save_maze', dest = 'save_maze', type = str, default = None,
                        help = 'save output to image file')
    parser.add_argument('--altcolor', dest = 'altcolor', default = False, action = 'store_true',
                        help = 'view in an alternate color scheme')

    args = parser.parse_args()
    main(args)
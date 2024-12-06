import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
        # TODO - MP11: Update the N-table. 

        # from __init__... we have our N-table (self.N)
        # whose last dimension corresponds to the 4 possible actions

        # action is a string that exists in self.actions which is a list

        # state is a tuple of 8 elems, each elem represents an index in the corresponding dim of self.N

        # if we do, then get the action's index off the index list based on the action
        action_index = self.actions.index(action)
        
        # Increment the N-table given the state and the type of action from the self.actions list
        # if already exists, will increment by +1 occurrence
        # if not, will init it for the state and action to a value of 1
        self.N[state][action_index] += 1

    def update_q(self, s, a, r, s_prime):
        # TODO - MP11: Update the Q-table. 

        # we want to update Q(s,a) --> the curr Q-value of the (state, action) pair
        # using the update rule: Q(s,a) = Q(s_t,a_t) + alpha * (r_t + gamma * maxQ(s_t+1, a) - Q(s_t, a_t))
        # r is the reward, alpha is the learning rate, gamma is the discount factor
        # for next state s we use the passed s_prime
        # for gamma we use self.gamma (__init__ function)
        # r is already passed in as r

        # let's convert the state to a tuple and get the action index so we don't have to keep doing it everytime we 
        # pass it in as indices in a Q or N table....
        s_tuple = tuple(s)
        s_prime_tuple = tuple(s_prime)
        action_index = self.actions.index(a)

        # 1) retrieve the current Q-value --> so we use the passed s, a as our indices (Q(s_t,a_t) part of the equation)
        q_value_curr = self.Q[s_tuple][action_index]

        # 2) calculate the max Q-value for the next state across all possible actions (maxQ(s_t+1, a) part of the equation)
        # calculate its max value using the Q-table and all actions as possibilities
        # use the numpy max function here to help
        next_q_val_max = np.max(self.Q[s_prime_tuple])
        
        # 3) compute learning rate (alpha part of the equation)
        # using the formula: alpha = C / (C + N(s,a))
        # N(s,a) is given by...
        n_s_a = self.N[s_tuple][action_index]
        # C is just given as self.C (from __init__)
        # plugging into given formula...
        alpha = self.C / (self.C + n_s_a)

        # 4) update q_value using the update rule: Q(s,a) = Q(s_t,a_t) + alpha * (r_t + gamma * maxQ(s_t+1, a) - Q(s_t, a_t))
        q_value = q_value_curr + alpha * (r + self.gamma * next_q_val_max - q_value_curr)

        # 5) finally, we store the new Q-value in our Q-table for the given (state, action)
        self.Q[s_tuple][action_index] = q_value

    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)

        # TODO - MP12: write your function here

        return utils.RIGHT

    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        '''
        # TODO - MP11: Implement this helper function that generates a state given an environment 

        # extract info from environment
        snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y = environment

        # 1) first, we want to identify the food direction relative to the snake head
        # we do this based on the relative position from the food position to the snake head position and populate the 3 possible values

        # for x (right/left) dir...
        if food_x > snake_head_x: # food on snake head right
            food_dir_x = 2 # dir is 2
        elif food_x < snake_head_x: # food on snake head left
            food_dir_x = 1 # dir is 1
        else: # same x coords on x axis
            food_dir_x = 0 # dir is 0

        # for y (up/down) dir...
        if food_y > snake_head_y: # food on snake head top
            food_dir_y = 2 # dir is 2
        elif food_y < snake_head_y: # food on snake head bottom
            food_dir_y = 1 # dir is 2
        else: # same coords on y axis
            food_dir_y = 0 # dir is 0

        # 2) next, we check if there are walls or rocks adjacent to the snake's head

        # for x (right/left) dir
        # wall / rock on snake head left OR wall / rock on both snake head left and right
        if snake_head_x == 1 or (snake_head_x == rock_x + 2 and snake_head_y == rock_y):
            adjoining_wall_x = 1 # set to 1
        # wall / rock on both snake head left and right
        elif snake_head_x == self.display_width - 2 or (snake_head_x == rock_x - 1 and snake_head_y == rock_y):
            adjoining_wall_x = 2 # set to 2
        # no adjoining wall/rock on x axis
        else:
            adjoining_wall_x = 0 # set to 0

        # for y (up/down)
        # wall / rock on snake head top or wall/rock on both snake head top and bottom
        if snake_head_y == 1 or (snake_head_y == rock_y + 1 and (snake_head_x == rock_x or snake_head_x == rock_x + 1)):
            adjoining_wall_y = 1 # set to 1
        # wall / rock on snake head bottom
        elif snake_head_y == rock_y - 1 and (snake_head_x == rock_x or snake_head_x == rock_x + 1) or snake_head_y == self.display_height - 2:
            adjoining_wall_y = 2 # set to 2
        # no adjoining wall / rock on y axis
        else:
            adjoining_wall_y = 0 # set to 0

        # 3) next, we check to see if a grid next to the snake head contains the snake body

        # adjoining top square has snake body
        # (x, y - 1)
        if (snake_head_x, snake_head_y - 1) in snake_body:
            adjoining_body_top = 1 # if so, 1
        # otherwise
        else:
            adjoining_body_top = 0 # 0
        
        # adjoining bottom square has snake body
        # (x, y + 1)
        if (snake_head_x, snake_head_y + 1) in snake_body:
            adjoining_body_bottom = 1 # if so, 1
        # otherwise
        else:
            adjoining_body_bottom = 0 # 0
        
        # adjoining left square has snake body
        # left is (x-1,y)
        if (snake_head_x - 1, snake_head_y) in snake_body:
            adjoining_body_left = 1 # if so, 1
        # otherwise
        else:
            adjoining_body_left = 0 # 0

        # adjoining right square has snake body
        # right is (x+1,y)
        if (snake_head_x + 1, snake_head_y) in snake_body:
            adjoining_body_right = 1 # if so, 1
        # otherwise
        else:
            adjoining_body_right = 0 # 0

        # return state tuple
        # format: Each state in the MDP is a tuple (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
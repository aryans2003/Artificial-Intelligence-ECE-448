import pygame
from pygame.locals import *
import argparse
import numpy as np

from agent import Agent
from snake import SnakeEnv
import utils
import time
import random
from test_data import actions_list, envs, discret_results

init_conds = [[5, 5, 2, 2, 18, 10, 1, 1],
              [5, 5, 2, 2, 18, 10, 3, 4],
              [5, 5, 2, 2, 10, 18, 3, 4]
              ]


class Application:
    def __init__(self, args):
        self.args = args
        self.env = SnakeEnv(args.snake_head_x, args.snake_head_y, args.food_x, args.food_y,
                            args.width, args.height, rock_x=args.rock_x, rock_y=args.rock_y)
        self.agent = Agent(self.env.get_actions(), args.Ne, args.C, args.gamma, args.width, args.height)

    def execute(self):
        if self.args.early_check:
            print("Early checking ... ")
            self.agent.train()
            match = True

            # check state discretization
            gold_results = np.array(discret_results)
            test_results = np.array([self.agent.generate_state(env) for env in envs])
            if not np.array_equal(test_results, gold_results):
                print("Warning: Your discretized states do not match gold ones")
                match = False
            else:
                print('State discretization tests passed!')

            # check Q-table and N-table
            for game in range(1, len(actions_list) + 1):
                print("Checking Game:", str(game) + "/" + str(len(actions_list)) + " with forced actions:",
                      actions_list[game - 1][:6], "...")
                self.env = SnakeEnv(self.args.snake_head_x, self.args.snake_head_y, self.args.food_x, self.args.food_y,
                                    self.args.width, self.args.height, rock_x=self.args.rock_x, rock_y=self.args.rock_y,
                                    random_seed=42)
                self.agent = Agent(self.env.get_actions(), self.args.Ne, self.args.C, self.args.gamma, self.args.width,
                                   self.args.height)

                # self.env.display()    # uncomment this line if you want to display the game
                # pygame.event.pump()   # uncomment this line if you want to display the game

                dead = False
                count = 0
                while not dead and count < len(actions_list[game - 1]):

                    environment = self.env.get_environment()
                    s_prime = self.agent.generate_state(environment)

                    if self.agent.s is not None and self.agent.a is not None:
                        self.agent.update_n(self.agent.s, self.agent.a)
                        self.agent.update_q(self.agent.s, self.agent.a, 0.1, s_prime)

                    # For debug convenience, the path to the gold tables are hardcoded here
                    # (see Debugging Examples section in spec)
                    action = actions_list[game - 1][count]
                    _, _, dead = self.env.step(action)
                    count += 1
                    self.agent.s = s_prime
                    self.agent.a = action

                # load gold table 
                gold_Q = np.load('./data/early_check/game_' + str(game) + '.npy')
                gold_N = np.load('./data/early_check/game_' + str(game) + '_N.npy')
                if not np.array_equal(self.agent.Q, gold_Q):
                    print("Warning: Your Q table does not match gold table for game", game)
                    match = False
                elif not np.array_equal(self.agent.N, gold_N):
                    print("Warning: Your N table does not match gold table for game", game)
                    match = False

            if match:
                print("Early checking passed!")
            else:
                print("Early checking failed!")

        else:
            if not self.args.human:
                if self.args.train_eps != 0:
                    self.train()
                self.test()
            self.show_games()

    def train(self):
        print("Train Phase:")
        self.agent.train()
        window = self.args.window
        self.points_results = []
        first_eat = True
        start = time.time()

        for game in range(1, self.args.train_eps + 1):
            environment = self.env.get_environment()
            dead = False
            action = self.agent.act(environment, 0, dead)
            while not dead:
                environment, points, dead = self.env.step(action)

                # For debug convenience, you can check if your Q-table mathches ours for given setting of parameters
                # (see Debugging Examples section in spec)
                if first_eat and points == 1:
                    self.agent.save_model(utils.CHECKPOINT)
                    first_eat = False

                action = self.agent.act(environment, points, dead)

            points = self.env.get_points()
            self.points_results.append(points)

            if game % self.args.window == 0:
                print(
                    f"Games: {len(self.points_results) - window} - {len(self.points_results)} \
                    Points (Average: {sum(self.points_results[-window:]) / window} \
                    Max: {max(self.points_results[-window:])} \
                    Min: {min(self.points_results[-window:])})"
                )

            self.env.reset()

        print(f"Training takes {time.time() - start:.2f} seconds")
        self.agent.save_model(self.args.model_name)

        return time.time() - start

    def test(self):
        print("Test Phase:")
        self.agent.eval()
        self.agent.load_model(self.args.model_name)
        start = time.time()
        all_points_results = []
        # (snake_head_x, snake_head_y, food_x, food_y, width, height, rock_x, rock_y)
        for i, cond in enumerate(init_conds):
            snake_head_x, snake_head_y, food_x, food_y, width, height, rock_x, rock_y = cond
            self.agent.display_width = width
            self.agent.display_height = height
            self.env = SnakeEnv(snake_head_x, snake_head_y, food_x, food_y, width, height, rock_x, rock_y)
            print('Testing init condition {}: snake_x: {} snake_y: {} food_x: {}, food_y: {}, '
                  'width: {}, height: {}, rock_x: {}, rock_y: {}'.format(
                i, snake_head_x, snake_head_y, food_x, food_y, width, height, rock_x, rock_y)
            )
            points_results = []
            for game in range(1, self.args.test_eps + 1):
                environment = self.env.get_environment()
                dead = False
                action = self.agent.act(environment, 0, dead)
                while not dead:
                    environment, points, dead = self.env.step(action)
                    action = self.agent.act(environment, points, dead)
                points = self.env.get_points()
                points_results.append(points)
                self.env.reset()

            print(f"Number of Games: {len(points_results)}")
            print(f"Average Points: {sum(points_results) / len(points_results)}")
            print(f"Max Points: {max(points_results)}")
            print(f"Min Points: {min(points_results)}")
            print(f"Testing takes {time.time() - start} seconds")
            all_points_results.append(sum(points_results) / len(points_results))
        return all_points_results

    def show_games(self):
        print("Display Games")
        self.env = SnakeEnv(self.args.snake_head_x, self.args.snake_head_y, self.args.food_x, self.args.food_y,
                            self.args.width, self.args.height, rock_x=self.args.rock_x, rock_y=self.args.rock_y)
        self.env.display()
        pygame.event.pump()
        self.agent.eval()
        points_results = []
        end = False
        for game in range(1, self.args.show_eps + 1):
            environment = self.env.get_environment()
            dead = False
            action = self.agent.act(environment, 0, dead)
            count = 0
            while not dead:
                count += 1
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                if keys[K_ESCAPE] or self.check_quit():
                    end = True
                    break
                environment, points, dead = self.env.step(action)
                # Qlearning agent
                if not self.args.human:
                    action = self.agent.act(environment, points, dead)
                # for human player
                else:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_UP:
                                action = 0
                            elif event.key == pygame.K_DOWN:
                                action = 1
                            elif event.key == pygame.K_LEFT:
                                action = 2
                            elif event.key == pygame.K_RIGHT:
                                action = 3
            if end:
                break
            self.env.reset()
            points_results.append(points)
            print("Game:", str(game) + "/" + str(self.args.show_eps), "Points:", points)
        if len(points_results) == 0:
            return
        print("Average Points:", sum(points_results) / len(points_results))

    def check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False


def main():
    parser = argparse.ArgumentParser(description='CS440 MP6 Snake')

    parser.add_argument('--human', default=False, action="store_true",
                        help='making the game human playable - default False')

    parser.add_argument('--early_check', default=False, action="store_true", )

    parser.add_argument('--model_name', dest="model_name", type=str, default="q_agent.npy",
                        help='name of model to save if training or to load if evaluating - default q_agent')

    parser.add_argument('--train_episodes', dest="train_eps", type=int, default=10000,
                        help='number of training episodes - default 10000')

    parser.add_argument('--test_episodes', dest="test_eps", type=int, default=1000,
                        help='number of testing episodes - default 1000')

    parser.add_argument('--show_episodes', dest="show_eps", type=int, default=10,
                        help='number of displayed episodes - default 10')

    parser.add_argument('--window', dest="window", type=int, default=100,
                        help='number of episodes to keep running stats for during training - default 100')

    parser.add_argument('--Ne', dest="Ne", type=int, default=40,
                        help='the Ne parameter used in exploration function - default 40')

    parser.add_argument('--C', dest="C", type=int, default=40,
                        help='the C parameter used in learning rate - default 40')

    parser.add_argument('--gamma', dest="gamma", type=float, default=0.7,
                        help='the gamma paramter used in learning rate - default 0.7')

    parser.add_argument('--snake_head_x', dest="snake_head_x", type=int, default=5,
                        help='initialized x position of snake head  - default 5')

    parser.add_argument('--snake_head_y', dest="snake_head_y", type=int, default=5,
                        help='initialized y position of snake head  - default 5')

    parser.add_argument('--food_x', dest="food_x", type=int, default=2,
                        help='initialized x position of food  - default 2')

    parser.add_argument('--food_y', dest="food_y", type=int, default=2,
                        help='initialized y position of food  - default 2')

    parser.add_argument('--width', type=int, default=18, help='The width of playing area - default 18')
    parser.add_argument('--height', type=int, default=10, help='The height of playing area - default 10')
    parser.add_argument('--rock_x', dest='rock_x', type=int, default=1, help='x position of the rock - default 1')
    parser.add_argument('--rock_y', dest='rock_y', type=int, default=1, help='y position of the rock - default 1')

    args = parser.parse_args()
    app = Application(args)
    app.execute()


if __name__ == "__main__":
    main()

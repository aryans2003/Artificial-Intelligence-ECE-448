import random
import pygame
import utils

LENGTH_FACTOR = 40 # for all lengths, multiply it with this factor for pygame display

class SnakeEnv:
    def __init__(self, snake_head_x, snake_head_y, food_x, food_y,
                 display_width, display_height, rock_x, rock_y, random_seed=None):
        self.game = Snake(snake_head_x, snake_head_y, food_x, food_y, display_width, display_height, rock_x, rock_y)
        self.render = False
        if random_seed:
            random.seed(random_seed)

    def get_actions(self):
        return self.game.get_actions()

    def reset(self):
        return self.game.reset()
    
    def get_points(self):
        return self.game.get_points()

    def get_environment(self):
        return self.game.get_environment()

    def step(self, action):
        environment, points, dead = self.game.step(action)
        if self.render:
            self.draw(environment, points, dead)
        return environment, points, dead

    def draw(self, environment, points, dead):
        snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y = environment
        self.display.fill(utils.BLUE)    
        pygame.draw.rect( self.display, utils.BLACK,
                [
                    LENGTH_FACTOR,
                    LENGTH_FACTOR,
                    (self.game.display_width -  2) * LENGTH_FACTOR,
                    (self.game.display_height - 2) * LENGTH_FACTOR
                ])
        
        # Draw rock
        pygame.draw.rect(
                    self.display, 
                    utils.GREY,
                    [
                        self.game.rock_x * LENGTH_FACTOR,
                        self.game.rock_y * LENGTH_FACTOR,
                        2 * LENGTH_FACTOR,
                        LENGTH_FACTOR
                    ]
                )

        # Draw snake head
        pygame.draw.rect(
                    self.display, 
                    utils.GREEN,
                    [
                        snake_head_x * LENGTH_FACTOR,
                        snake_head_y * LENGTH_FACTOR,
                        LENGTH_FACTOR,
                        LENGTH_FACTOR
                    ],
                    3
                )

        # Draw snake body
        for seg in snake_body:
            pygame.draw.rect(
                self.display, 
                utils.GREEN,
                [
                    seg[0] * LENGTH_FACTOR,
                    seg[1] * LENGTH_FACTOR,
                    LENGTH_FACTOR,
                    LENGTH_FACTOR,
                ],
                1
            )

        # Draw food
        pygame.draw.rect(
                    self.display, 
                    utils.RED,
                    [
                        food_x * LENGTH_FACTOR,
                        food_y * LENGTH_FACTOR,
                        LENGTH_FACTOR,
                        LENGTH_FACTOR
                    ]
                )

        text_surface = self.font.render("Points: " + str(points), True, utils.BLACK)
        text_rect = text_surface.get_rect()
        text_rect.center = ((280),(25))
        self.display.blit(text_surface, text_rect)
        pygame.display.flip()
        if dead:
            # slow clock if dead
            self.clock.tick(1)
        else:
            self.clock.tick(5)

        return 

    def display(self):
        pygame.init()
        pygame.display.set_caption('MP6: Snake')
        self.clock = pygame.time.Clock()
        pygame.font.init()

        self.font = pygame.font.Font(pygame.font.get_default_font(), 15)
        self.display = pygame.display.set_mode((self.game.display_width * LENGTH_FACTOR, self.game.display_height * LENGTH_FACTOR), pygame.HWSURFACE)
        self.draw(self.game.get_environment(), self.game.get_points(), False)
        self.render = True
            
class Snake:
    def __init__(self, snake_head_x, snake_head_y, food_x, food_y, display_width, display_height, rock_x, rock_y):
        self.init_snake_head_x = snake_head_x
        self.init_snake_head_y = snake_head_y
        self.init_food_x = food_x
        self.init_food_y = food_y
        self.display_width = display_width
        self.display_height = display_height
        self.rock_x = rock_x
        self.rock_y = rock_y
        
        # This quantity mentioned in the spec, 8*16*8
        self.starve_steps = 8 * (self.display_height) * (self.display_width)
        self.reset()

    def reset(self):
        self.points = 0
        self.steps = 0
        self.snake_head_x = self.init_snake_head_x
        self.snake_head_y = self.init_snake_head_y
        self.snake_body = []
        self.food_x = self.init_food_x
        self.food_y = self.init_food_y

    def get_points(self):
        # These points only updated when food eaten
        return self.points

    def get_actions(self):
        # Corresponds to up, down, left, right
        # return [0, 1, 2, 3]
        return utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

    def get_environment(self):
        return [
            self.snake_head_x,
            self.snake_head_y,
            self.snake_body,
            self.food_x,
            self.food_y, 
            self.rock_x,
            self.rock_y
        ]

    def step(self, action):
        is_dead = self.move(action)
        return self.get_environment(), self.get_points(), is_dead

    def move(self, action):
        self.steps += 1

        delta_x = delta_y = 0

        # Up
        if action == utils.UP:
            delta_y = -1 
        # Down
        elif action == utils.DOWN:
            delta_y = 1
        # Left
        elif action == utils.LEFT:
            delta_x = -1 
        # Right
        elif action == utils.RIGHT:
            delta_x = 1

        old_body_head = None
        if len(self.snake_body) == 1:
            old_body_head = self.snake_body[0]

        # Snake "moves" by 1. adding previous head location to body, 
        self.snake_body.append((self.snake_head_x, self.snake_head_y))
        # 2. updating new head location via delta_x/y
        self.snake_head_x += delta_x
        self.snake_head_y += delta_y

        # 3. removing tail if body size greater than food eaten (points)
        if len(self.snake_body) > self.points:
            del(self.snake_body[0])

        # Eats food, updates points, and randomly generates new food, if appropriate
        self.handle_eatfood()

        # Colliding with the snake body or going backwards while its body length
        # greater than 1
        if len(self.snake_body) >= 1:
            for seg in self.snake_body:
                if self.snake_head_x == seg[0] and self.snake_head_y == seg[1]:
                    return True

        # Moving towards body direction, not allowing snake to go backwards while 
        # its body length is 1
        if len(self.snake_body) == 1:
            if old_body_head == (self.snake_head_x, self.snake_head_y):
                return True

        # Check if collide with the wall (left, top, right, bottom)
        # Again, views grid position wrt top left corner of square
        if (self.snake_head_x < 1 or self.snake_head_y < 1 or
            self.snake_head_x + 1 > self.display_width-1 or self.snake_head_y + 1 > self.display_height-1):
            return True
            
        # Check if collide with the rock
        if ((self.snake_head_x == self.rock_x or self.snake_head_x == self.rock_x + 1) and self.snake_head_y == self.rock_y):
            return True
        
        # looping for too long and starved
        if self.steps > self.starve_steps:
            return True

        return False

    def handle_eatfood(self):
        if (self.snake_head_x == self.food_x) and (self.snake_head_y == self.food_y):
            self.random_food()
            self.points += 1
            self.steps = 0

    def random_food(self):

        # Math looks at upper left corner wrt DISPLAY_SIZE for grid coordinates

        max_x = self.display_width - 2
        max_y = self.display_height - 2
        
        self.food_x = random.randint(1, max_x)
        self.food_y = random.randint(1, max_y) 
        
        while self.check_food_on_rock() or self.check_food_on_snake():
            self.food_x = random.randint(1, max_x)
            self.food_y = random.randint(1, max_y)
    
    def check_food_on_rock(self):
        return ((self.food_x == self.rock_x or self.food_x == self.rock_x + 1)
               and self.food_y == self.rock_y)
               
    def check_food_on_snake(self):
        if self.food_x == self.snake_head_x and self.food_y == self.snake_head_y:
            return True 
        for seg in self.snake_body:
            if self.food_x == seg[0] and self.food_y == seg[1]:
                return True
        return False
        
    

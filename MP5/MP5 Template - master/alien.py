# alien.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by James Gao (jamesjg2@illinois.edu) on 9/03/2021
import numpy as np

"""
This file contains code to represent the alien that we are using as our principal actor for MP5/6
"""


class Alien:
    """The Meaty Alien that will be navigating our map

        The alien has two forms that are represented with geometric shapes:

        Form 1 (Meatball):
            A circle with a fixed radius.

        Form 2 (Sausage):
            An oblong (sausage shape). This is represented as a line segment with a fixed length, and
            The width of the alien's entire sausage body is the diameter of these circles.
    """

    def __init__(self, centroid, lengths, full_shape_widths, shapes, init_shape, window):

        self.__centroid = centroid  # centroid of the alien (x,y)
        self.__full_shape_width = full_shape_widths # widths of the objects in each shape, in this case (line width, diameter, line width) for (Horizontal,Ball,Vertical)
        self.__lengths = lengths  # lengths of the line segment in each shape (line length, 0, line length) for (Horizontal, Ball, Vertical).
        self.__shapes = shapes  # possible shapes that the alien can have, in this case ('Horizontal','Ball','Vertical')
        self.__shape = init_shape
        self.__shapenum = self.__shapes.index(self.__shape)
        self.__limits = [[0, window[0]], [0, window[1]], [0, len(self.__shapes)]]

    # Returns a tuple with the (x,y) coordinates of the alien's head and tail ((x_head,y_head), (x_tail,y_tail))
    def get_head_and_tail(self):
        length = self.get_length() / 2
        if self.__shape == 'Horizontal':
            head = (self.__centroid[0] + length, self.__centroid[1])
            tail = (self.__centroid[0] - length, self.__centroid[1])
        elif self.__shape == 'Vertical':
            head = (self.__centroid[0], self.__centroid[1] - length)
            tail = (self.__centroid[0], self.__centroid[1] + length)
        elif self.__shape == 'Ball':
            head = (self.__centroid[0], self.__centroid[1])
            tail = head
        else:
            raise ValueError('Invalid shape!')
        return head, tail

    # Returns the centroid position of the alien
    def get_centroid(self):
        return self.__centroid

    # Returns length of the line segment in the current form of the alien
    def get_length(self):
        return self.__lengths[self.__shapenum]

    # Returns the radius of the current shape (the half of the width of the full shape)
    def get_width(self):
        return self.__full_shape_width[self.__shapenum] / 2

    # Returns whether the alien is in circle or oblong form. True is alien is in circle form, False if oblong form.
    def is_circle(self):
        return self.__shape == 'Ball'

    def set_alien_pos(self, pos):
        """Sets the alien's centroid position to the specified pos argument.

            Args:
                pos: The (x,y) coordinate position we want to place the alien's centroid
        """
        self.__centroid = pos

    def set_alien_shape(self, shape):
        """Sets the alien's shape to the specified shape argument.

            Args:
                shape: str. The alien's shape we want to set
        """
        if (np.abs(self.__shapes.index(shape) - self.__shapenum) <= 1) and (shape in self.__shapes):
            self.__shape = shape
            self.__shapenum = self.__shapes.index(self.__shape)
        else:
            # raise exception for illegal transformation
            raise ValueError("Illegal alien transformation.")

    def set_alien_config(self, config):
        self.__centroid = [config[0], config[1]]
        self.__shape = config[2]
        self.__shapenum = self.__shapes.index(self.__shape)

    def get_shape_idx(self):
        return self.__shapenum

    def get_alien_limits(self):
        return self.__limits

    def get_config(self):
        return [self.__centroid[0], self.__centroid[1], self.__shape]

    def get_shapes(self):
        return self.__shapes

    def get_shape(self):
        return self.__shape

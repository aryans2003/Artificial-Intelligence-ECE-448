# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy
import math # import math for sqrt()


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """

    #1) check if alien is circle or oblong
    # from alien.py is_circle() -- True is alien is in circle form, False if oblong form.
    if alien.is_circle():
        # for a circular alien, extract center and radius using given helper functions
        center = alien.get_centroid()
        radius = alien.get_width()
        # for each 'list of endpoints' in given format...
        for startx, starty, endx, endy in walls:
            # convert to segment
            segment = ((startx, starty), (endx, endy))
            # calculate distance from center to wall segment using point_segment_distance() helper funct
            distance = point_segment_distance(center, segment)
            # if this distance <= radius
            if distance <= radius:
                return True # alien touches the wall
    # else, our alien must be in oblong form
    else:
        # extract head and tail coordinates -- this is our line segment
        # Returns a tuple with the (x,y) coordinates of the alien's head and tail ((x_head,y_head), (x_tail,y_tail))
        head, tail = alien.get_head_and_tail()
        alien_segment = (head, tail)
        # extract width
        radius = alien.get_width()
         # for each 'list of endpoints' in given format...
        for startx, starty, endx, endy in walls:
            # convert to segment
            segment = ((startx, starty), (endx, endy))
            # calc distance between line segment and wall segment using segment_distance() helper funct
            distance = segment_distance(alien_segment, segment)
            # if distance <= radius
            if distance <= radius:
                return True # alien touches the wall
    # if all else fails
    return False

def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """

    # extract dimensions
    width_window, height_window = window

    # get position (centroid), radius (width / 2), length using helper functions
    center_x, center_y = alien.get_centroid()
    width = alien.get_width()
    radius = width / 2
    length = alien.get_length()

    # check alien shape
    if alien.is_circle():
        # if circle, alculate extreme points
        leftmost = center_x - radius
        rightmost = center_x + radius
        topmost = center_y - radius
        bottommost = center_y + radius
        # check if within window boundaries (left >= 0, right <= window width, topmost >= 0, bottommost <= window height)
        if(leftmost > 0 and rightmost < width_window and topmost > 0 and bottommost < height_window):
            return True # if so, return true
    # alien shape is oblong
    else:
        # if shape is Horizontal
        if alien.get_shape() == "Horizontal":
            # calculate extreme points by using length / 2  in addition to radius
            leftmost = center_x - (length / 2)
            rightmost = center_x + (length / 2)
            topmost = center_y - radius
            bottommost = center_y + radius
        # if shape is not Horizontal
        else:
            # calculate extreme points by using length / 2  in addition to radius, flip dimensions
            leftmost = center_x - radius
            rightmost = center_x + radius
            topmost = center_y - (length / 2)
            bottommost = center_y + (length / 2)
        # check if within window boundaries
        if (leftmost > 0 and rightmost < width_window and topmost > 0 and bottommost < height_window):
            return True # if so, return true

    return False # if all else fails...


def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """

    # extract vertices
    (x1, y1) = polygon[0] # A
    (x2, y2) = polygon[1] # B
    (x3, y3) = polygon[2] # C
    (x4, y4) = polygon[3] # D
    (x5, y5) = point

    # calculate areas of triangles formed by the point and with the vertices of the parallelogram
    # for triangles ABP, BCP, CDP, DAP
    ABP = calc_triangle_area((x1, y1), (x2, y2), (x5, y5))
    BCP = calc_triangle_area((x2, y2), (x3, y3), (x5, y5))
    CDP = calc_triangle_area((x3, y3), (x4, y4), (x5, y5))
    DAP = calc_triangle_area((x4, y4), (x1, y1), (x5, y5))

    # calculate area of parallelogram
    # A = (1/2) * |x1y2 + x2y3 + x3y4 + x4y1 - (y1x2 + y2x3 + y3x3 + y3x4 + y4x1)|
    area = (1 / 2) * abs( (x1 * y2) + (x2 * y3) + (x3 * y4) + (x4 * y1) - (y1 * x2) - (y2 * x3) - (y3 * x4) - (y4 * x1) ) 

    # if sum of areas of triangles == parallelogram area
    if (ABP + BCP + CDP + DAP) == area:
        return True # P is inside parallelogram

    return False # else, not

# helper function to claculate area of a triangle defined by 3 passed points
# points are passed as tuples (x,y)
def calc_triangle_area(point1, point2, point3):

    # extract points
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    x3 = point3[0]
    y3 = point3[1]

    # Area = (1/2) * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    return (1/2) * abs( (x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)) )


def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """

    # 1) extract current position of alien
    # starting point -- centroid position
    starting = alien.get_centroid()

    # 2) create a line segment from alien's current position to waypoint
    segment = (starting, waypoint)

        # Calculate the diagonal of the alien's bounding box
    diag = np.sqrt(alien.get_width()**2 + alien.get_length()**2)

    #3) check each wall to see if intersects with our segment in given format
    for startx, starty, endx, endy in walls:
        # convert to wall
        wall = ((startx, starty), (endx, endy))
        # check if segments intersect using helper function
        if do_segments_intersect(segment, wall):
            return True # alien path does touch wall
                # Check if the distance from the path to the wall is less than the alien's dimensions

        distance = segment_distance(segment, wall)
        if distance < diag / 2: 
            return True  # Path is too close to a wall
    # after all walls checked, all else fails
    return False

def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """

    # in this function, want to calculate the Euclidean distance from a point to a line segment

    # 1) Convert segment into Ax + By + C = 0 format
    # extract line segment tuple
    (x1, y1), (x2, y2) = s
    # A = y2 - y1
    A = y2 - y1
    # B = x1 - x2
    B = x1 - x2
    # C = (x2*y1) - (x1*y2)
    C = (x2 * y1) - (x1 * y2)

    # 2) extract coordinate point tuple
    (x, y) = p

    # 3) check whether point's projection falls onto line segment
    # calc dot product
    product = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
    # calculate the square of the line segment's length
    length = (x2 - x1) ** 2 + (y2 - y1) ** 2
    # check if projection is within segment
    if product <= 0:
        # p is closer to (x1, y1)
        # compute Euclidean distance -- sqrt((x2 - x1)^2) + (y2 - y1)^2)
        return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    elif product >= length:
        # p is closer to (x2, y2)
        # compute Euclidean distance
        return math.sqrt((x - x2) ** 2 + (y - y2) ** 2)

    # 4) else, return perpendicular distance d = ( | (A*x) + (B*y) + C | ) / sqrt(A^2 + B^2) 
    return (abs((A*x) + (B*y) + C)) / math.sqrt(A**2 + B**2) 


def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """

    # might need to handle collinear segments

    # extracting points from s1, s2
    (x1, y1) = s1[0]
    (x2, y2) = s1[1]
    (x3, y3) = s2[0]
    (x4, y4) = s2[1]

    # calculate dx, dy
    dx_s1 = x2 - x1
    dy_s1 = y2 - y1
    dx_s2 = x4 - x3
    dy_s2 = y4 - y3

    # calculate cross product
    cross_product = (dx_s1 * dy_s2) - (dy_s1 * dx_s2)

    # handling collinear segments
    # if cross_product is 0, lines are parallel or collinear
    if abs(cross_product) == 0:
        # check if endpoint of a segment lines on the other segment
        # using defined helper function
        if on_segment(s1[0], s2[0], s1[1]) or on_segment(s1[0], s2[1], s1[1]) or on_segment(s2[0], s1[0], s2[1]) or on_segment(s2[0], s1[1], s2[1]):
            return True # collinear and overlapping
        return False # parallel and not overlapping
    
    # else, we calculate individual parameters
    # the parameters represent intersection points along the segment
    t = ((x3 - x1) * dy_s2 - (y3 - y1) * dx_s2) / cross_product
    u = ((x3 - x1) * dy_s1 - (y3 - y1) * dx_s1) / cross_product

    # if both t and u are between [0,1] (<-- inclusive) then the lines intersect
    if 0 <= t <= 1 and 0 <= u <= 1:
        return True

    return False # if all else fails

# helper function to check if a point q is on segment pr
# where pr is the first and second endpoints of the segment, respectively
# and q is the point we want to check
def on_segment(p, q, r):
    # if q's x-coordinate is between p's and r's x-coords
    # if q's y-coordinate is between p's and r's y-coords
    # if both conditions satisfied, q is on segment pr
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))


def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """

    # we can use the functions we defined above as helper functs

    # check if they intersect
    if do_segments_intersect(s1, s2):
        return 0 # if they intersect, distance is simply 0
    
    # if they don't intersect, there must be a nonzero distance
    # store distances in an array using point_segment_distance for all points (x, y) for each s1, s2
    distances = [point_segment_distance(s1[0], s2), point_segment_distance(s1[1], s2), point_segment_distance(s2[0], s1), point_segment_distance(s2[1], s1)]

    # return min val
    return min(distances)


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")

import numpy as np


# class task to add to environment so tasks can be interleaved in the same environment
class Task:
    """Base class for tasks.

    Args:
        ...
    """

    def __init__(
        self,
        name: str = "Task",
        goal_locations=None,
        start_coord_x=0,
        start_coord_y=None,
    ):
        self.__name__ = name
        self.goal_locations = goal_locations
        # self.start_locations = start_locations
        self.start_coord_x = start_coord_x
        self.start_coord_y = start_coord_y  # TODO: if start coord for hand is none, find midpoint of goals


# helper for line goal tasks
def get_line_miller_points(scale=0.011, lift_height=0.4):
    # scale relative to original point list from experiment
    # lift height is the y coordinate of the points

    # original points from the miller experiment
    original_point_list = [-11, -9, -7, 7, 9, 11]

    new_list = [x * scale for x in original_point_list]

    # add lift height as y coordinate
    points = [[x, lift_height] for x in new_list]

    return np.array(points)


class OneDimensionalReach(Task):
    """Centre-out reach task (1D).

    Args:
        ...

    """

    def __init__(
        self,
        name: str = "OneDimensionalReach",
        scale=0.011,
        # lift_height=0.4,
        start_coord_x=0,
        start_coord_y=0.4,
    ):

        # self.scale =scale

        goal_locations = get_line_miller_points(scale, start_coord_y)

        start_coord_x = start_coord_x
        # set start_coord_y with default lift height if none is provided
        # start_coord_y = start_coord_y

        super().__init__(name, goal_locations, start_coord_x, start_coord_y)


# helper for circle goal tasks
# # get coordinates of 8 points evenly spaced around a circle
def get_circle_points(radius=0.1, n_points=8, lift_height=0.4):
    points = []
    for i in range(n_points):
        angle = i * 2 * np.pi / n_points
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        if lift_height != 0.0:
            y += lift_height
        points.append([x, y])
    return np.array(points)


class CentreOutReach(Task):
    """Centre-out reach task.

    Args:
        ...
    """

    def __init__(
        self,
        name: str = "CentreOutReach",
        # lift_height=0.4,
        start_coord_x=0,
        start_coord_y=0.4,
        radius=0.1,
        n_points=8,
    ):
        goal_locations = get_circle_points(radius, n_points, start_coord_y)

        super().__init__(name, goal_locations, start_coord_x, start_coord_y)

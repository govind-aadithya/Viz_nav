from enum import Flag, Enum
import numpy as np

"""
DO NOT MODIFY THIS FILE!
This file defines the interface between the server code and the player's code
"""


class Action(Flag):
    IDLE = 0
    # movement actions
    FORWARD = 1
    BACKWARD = 2
    LEFT = 4
    RIGHT = 8
    # check-in when player believes a target is reached
    CHECKIN = 16
    # end the current phase
    QUIT = 32


class Phase(Enum):
    EXPLORATION = 1
    NAVIGATION = 2


class Player:
    """
    This is a base class. Inherit it like the KeyboardPlayerPyGame example in player.py
    """
    def __init__(self):
        self._targets = None  # this is to be set by the simulation server when it is being constructed
        self._state = None  # this is to be set by the simulation server
        self.reset()

    def reset(self) -> None:
        """
        This function is to be invoked by the simulation play function before starting the game
        :return: None
        """
        raise NotImplementedError('Your player class should at least implement '
                                  'this function to enable reset of the players.')

    def before_exploration(self) -> None:
        print('before exploration')

    def after_exploration(self) -> None:
        print('after exploration')

    def before_navigation(self) -> None:
        print('before navigation')

    def after_navigation(self) -> None:
        print('after navigation')

    def act(self) -> Action:
        """
        This function is to be invoked by the simulation server in each step, right after invoking see(fpv)

        return an action
        """
        raise NotImplementedError('Your player class should at least implement '
                                  'this function to tell the robot what to do next after seeing the image.')

    def see(self, fpv: np.ndarray) -> None:
        """
        This function is to be invoked by the simulation server in each step
        :param fpv: an opencv image (BGR format)
        """
        raise NotImplementedError('Your player class should at least implement '
                                  'this function to receive a new observation.')

    def get_target_images(self) -> list[np.ndarray]:
        """
        This function is to be invoked by players
        :return: a reference to the internal list of target fpv images, i.e., self._targets
        """
        return self._targets

    def set_target_images(self, images: list[np.ndarray]) -> None:
        """
        This function is to be invoked by the simulation server after exploration is finished
        :param images: a list of images that represents the target
        :return: None
        """
        self._targets = images

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state
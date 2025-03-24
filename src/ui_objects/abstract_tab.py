"""
This module defines the abstract base classes for tab widgets used in the Depth Estimator application.
"""

from abc import ABC, abstractmethod

from PyQt5 import QtWidgets

class CommonMeta(type(ABC), type(QtWidgets.QWidget)):
    """
    A metaclass that combines the metaclasses of QWidget and ABC.
    """

class AbstractTabWidget(ABC):
    """
    An abstract base class for tab widgets in the main window.
    """

    def __init__(self):
        """
        Initialize the AbstractTabWidget.
        """

    @property
    @abstractmethod
    def width(self) -> int:
        """
        Get the width of the camera image.

        Returns
        -------
        float
            The width of the camera image.
        """
        raise NotImplementedError

    @width.setter
    @abstractmethod
    def width(self, new_width: int):
        """
        Set the width of the camera image.

        Parameters
        ----------
        value : int
            The width of the camera image.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def height(self) -> int:
        """
        Get the height of the camera image.

        Returns
        -------
        float
            The height of the camera image.
        """
        raise NotImplementedError

    @height.setter
    @abstractmethod
    def height(self, new_height: int):
        """
        Set the height of the camera image.

        Parameters
        ----------
        value : int
            The height of the camera image.
        """
        raise NotImplementedError

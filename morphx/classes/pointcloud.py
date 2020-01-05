# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
from scipy.spatial.transform import Rotation as Rot


class PointCloud(object):
    """
    Class which represents a collection of points in 3D space. The points could have labels.
    """

    def __init__(self, vertices: np.ndarray, labels: np.ndarray = None, encoding: dict = None):
        """
        Args:
            vertices: Point coordinates with shape (n, 3).
            labels: Vertex label array with shape (n, 1).
            encoding: Dict with unique labels as keys and description string for respective label as value.
        """
        if vertices.shape[1] != 3:
            raise ValueError("Vertices must have shape (N, 3).")
        self._vertices = vertices

        if labels is None:
            self._labels = np.ndarray([])
        if labels is not None:
            if len(labels) != len(vertices):
                raise ValueError("Vertex label array must have same length as vertices array.")
            self._labels = labels.reshape(len(labels), 1)

        self._encoding = encoding
        self._class_num = len(np.unique(labels))

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def encoding(self) -> dict:
        return self._encoding

    @property
    def class_num(self) -> int:
        return self._class_num

    @property
    def weights_mean(self) -> np.ndarray:
        """ Extract frequences for each class and calculate weights as frequences.mean() / frequences, ignoring any
        labels which don't appear in the dataset (setting their weight to 0).

        Returns:
            np.ndarray with weights for each class.
        """

        if len(self._labels) != 0:
            total_labels = self._labels
            non_zero = []
            freq = []
            for i in range(self._class_num):
                freq.append((total_labels == i).sum())
                if freq[i] != 0:
                    # save for mean calculation
                    non_zero.append(freq[i])
                else:
                    # prevent division by zero
                    freq[i] = 1
            mean = np.array(non_zero).mean()
            freq = mean / np.array(freq)
            freq[(freq == mean)] = 0
            return freq
        else:
            return np.array([])

    @property
    def weights_occurence(self) -> np.ndarray:
        """ Extract frequences for each class and calculate weights as len(vertices) / frequences.

        Returns:
            np.ndarray with weights for each class.
        """

        if len(self._labels) != 0:
            class_num = self._class_num
            total_labels = self._labels
            freq = []
            for i in range(class_num):
                freq.append((total_labels == i).sum())
            freq = len(total_labels) / np.array(freq)
            return freq
        else:
            return np.array([])

    # -------------------------------------- TRANSFORMATIONS ------------------------------------------- #

    def normalize(self, radius: int):
        """ Divides the coordinates of the points by the context size (e.g. radius of the local BFS). If radius is not
            valid (<= 0) it gets set to 1, so that the normalization has no effect. """

        if radius <= 0:
            radius = 1
        self._vertices = self._vertices / radius

    def rotate_randomly(self, angle_range: tuple = (-180, 180)):
        """ Randomly rotates the vertices by performing an Euler rotation. The three angles are choosen randomly
            from the given angle_range. """

        # switch limits if lower limit is larger
        if angle_range[0] > angle_range[1]:
            angle_range = (angle_range[1], angle_range[0])

        angles = np.random.uniform(angle_range[0], angle_range[1], (1, 3))[0]
        r = Rot.from_euler('xyz', angles, degrees=True)
        if len(self._vertices) > 0:
            self._vertices = r.apply(self._vertices)

    def center(self):
        """ Centers the vertices around the centroid of the vertices. """

        centroid = np.mean(self._vertices, axis=0)
        self._vertices = self._vertices - centroid

    def add_noise(self, limits: tuple = (-1, 1)):
        """ Adds some random variation (amplitude given by the limits parameter) to the vertices. """

        # switch limits if lower limit is larger
        if limits[0] > limits[1]:
            limits = (limits[1], limits[0])
        # do nothing if limits are the same
        if limits[0] == limits[1]:
            return

        variation = np.random.random(self._vertices.shape) * (limits[1] - limits[0]) + limits[0]
        self._vertices = self._vertices + variation

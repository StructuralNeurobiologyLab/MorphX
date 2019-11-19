# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np


class PointCloud(object):
    """
    Class which represents a collection of points in 3D space. The points could have labels.
    """

    def __init__(self, vertices: np.ndarray, labels=None):
        """
        Args:
            vertices: coordinates of the mesh vertices which surround the skeleton with shape (n, 3).
            labels: vertex label array with same shape as vertices.
        """
        self._vertices = vertices

        self._labels = None
        if labels is not None:
            self._labels = labels

    @property
    def vertices(self) -> np.ndarray:
        """ Coordinates of the mesh vertices which surround the skeleton as np.ndarray
        with shape (n, 3).
        """
        return self._vertices

    @property
    def labels(self) -> np.ndarray:
        """ Vertex label array with same shape as vertices. """
        return self._labels

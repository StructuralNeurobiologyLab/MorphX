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
            vertices: point coordinates with shape (n, 3).
            labels: vertex label array with same shape as vertices.
        """
        if vertices.shape[1] != 3:
            raise ValueError("Vertices must have shape (N, 3).")
        self._vertices = vertices

        self._labels = None
        if labels is not None:
            if len(labels) != len(vertices):
                raise ValueError("Labels array must have same length as vertices array.")
            self._labels = labels

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    def set_vertices(self, vertices) -> None:
        if vertices.shape != self._vertices.shape:
            raise ValueError("Shape of vertices must not change as labels would loose their reference.")
        self._vertices = vertices

    @property
    def labels(self) -> np.ndarray:
        return self._labels

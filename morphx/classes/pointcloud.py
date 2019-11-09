# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.spatial import cKDTree


class PointCloud(object):
    """
    Class which represents a collection of points in 3D space.
    """

    def __init__(self, vertices: np.ndarray):
        """
        Args:
            vertices: coordinates of the mesh vertices which surround the skeleton with shape (n, 3).
        """
        self._vertices = vertices

    @property
    def vertices(self) -> np.ndarray:
        """ Coordinates of the mesh vertices which surround the skeleton as np.ndarray
        with shape (n, 3).
        """
        return self._vertices

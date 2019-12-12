# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
from morphx.classes.pointcloud import PointCloud


class MeshCloud(PointCloud):
    """ Class which represents a mesh with possible labels """
    def __init__(self,
                 vertices: np.ndarray,
                 faces: np.ndarray,
                 normals: np.ndarray,
                 labels: np.ndarray = None,
                 encoding: dict = None):
        """
        Args:
            vertices: Coordinates of the mesh vertices which surround the skeleton with shape (n, 3).
            faces: The faces of the mesh as array of the respective vertices with shape (n, 3).
            normals: The normal vectors of the mesh.
            labels: Vertex label array (integer number representing respective classe) with same dimensions as
                vertices.
            encoding: Dict with unique labels as keys and description string for respective label as value.
        """

        super().__init__(vertices, labels=labels, encoding=encoding)

        self._faces = faces
        self._normals = normals

    @property
    def faces(self) -> np.ndarray:
        return self._faces

    @property
    def normals(self) -> np.ndarray:
        return self._normals

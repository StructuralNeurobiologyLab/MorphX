# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch, Philipp Schubert

import numpy as np
import numba as nb
from collections import defaultdict
from scipy.spatial import cKDTree
from morphx.classes.hybridcloud import HybridCloud


class HybridMesh(HybridCloud):
    """ Class which represents a skeleton in form of a graph structure and a mesh which surrounds this skeleton. """

    def __init__(self,
                 nodes: np.ndarray,
                 edges: np.ndarray,
                 vertices: np.ndarray,
                 faces: np.ndarray,
                 normals: np.ndarray,
                 vert2skel: defaultdict = None,
                 labels: np.ndarray = None,
                 encoding: dict = None):
        """
        Args:
            nodes: Coordinates of the nodes of the skeleton with shape (n, 3).
            edges: Edge list with indices of nodes in skel_nodes with shape (n, 2).
            vertices: Coordinates of the mesh vertices which surround the skeleton with shape (n, 3).
            faces: The faces of the mesh as array of the respective vertices with shape (n, 3).
            normals: The normal vectors of the mesh.
            vert2skel: Dict structure that maps mesh vertices to skeleton nodes. Keys are skeleton node indices,
                values are lists of mesh vertex indices.
            labels: Vertex label array (integer number representing respective classe) with same dimensions as
                vertices.
            encoding: Dict with unique labels as keys and description string for respective label as value.
        """
        super().__init__(nodes, edges, vertices, vert2skel, labels=labels, encoding=encoding)

        self._faces = faces
        self._normals = normals
        self._faces2node = None

    @property
    def faces(self) -> np.ndarray:
        return self._faces

    @property
    def normals(self) -> np.ndarray:
        return self._normals

    @property
    def faces2node(self) -> dict:
        # TODO: slow, optimization required.
        """ Creates python dict with indices of ``:py:attr:~nodes`` as
        keys and lists of face indices associated with the nodes as values.
        All faces which contain at least ONE "node"-vertex are associated with
        this node.

        Returns:
            Python dict with mapping information.
        """
        if self._faces2node is None:
            self._faces2node = dict()
            sh = self.faces.shape
            for node_ix, vert_ixs in self.vert2skel.items():
                if len(vert_ixs) == 0:
                    self._faces2node[node_ix] = []
                    continue
                new_faces = any_in_1d_nb(self.faces, set(vert_ixs))
                new_faces = np.nonzero(new_faces)[0].tolist()
                self._faces2node[node_ix] = new_faces
        return self._faces2node

    @property
    def faces2node_OLD(self) -> dict:
        """ Creates python defaultdict with indices of ``:py:attr:~nodes`` as
        keys and lists of face indices associated with the nodes as values.

        Returns:
            Python defaultdict with mapping information.
        """
        self._faces2node = dict()
        for node_ix, vert_ixs in self.vert2skel.items():
            new_faces = np.all(np.isin(self.faces, vert_ixs), axis=1)
            self._faces2node[node_ix] = new_faces
        return self._faces2node


@nb.njit(parallel=True)
def any_in_1d_nb(matrix, indices):
    out = np.empty(matrix.shape[0], dtype=nb.boolean)
    for i in nb.prange(matrix.shape[0]):
        if (matrix[i, 0] in indices) or (matrix[i, 1] in indices) or (matrix[i, 2] in indices):
            out[i] = True
        else:
            out[i] = False
    return out

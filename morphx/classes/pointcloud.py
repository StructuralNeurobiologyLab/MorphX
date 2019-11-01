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

    def __init__(self, skel_nodes: np.ndarray, skel_edges: np.ndarray, vertices: np.ndarray):
        """
        Args:
            skel_nodes: coordinates of the nodes of the skeleton with shape (n, 3).
            skel_edges: edge list with indices of nodes in skel_nodes with shape (n, 2).
            vertices: coordinates of the mesh vertices which surround the skeleton with shape (n, 3).
        """
        self._skel_nodes = skel_nodes
        self._skel_edges = skel_edges
        self._vertices = vertices

        self._mesh2skel_dict = None
        self._weighted_graph = None
        self._simple_graph = None

    @property
    def skel_nodes(self) -> np.ndarray:
        """ Coordinates of the nodes of the skeleton.
        """
        return self._skel_nodes

    @property
    def skel_edges(self) -> np.ndarray:
        """ Edge list with indices of nodes in skel_nodes as np.ndarray with shape (n, 2).
        """
        return self._skel_edges

    @property
    def vertices(self) -> np.ndarray:
        """ Coordinates of the mesh vertices which surround the skeleton as np.ndarray
        with shape (n, 3).
        """
        return self._vertices

    def mesh2skel_dict(self) -> defaultdict:
        """ Creates python defaultdict with indices of skel_nodes as keys and lists of mesh vertex
        indices which have their key node as nearest skeleton node.

        Returns:
            Python defaultdict with mapping information
        """
        if self._mesh2skel_dict is None:
            tree = cKDTree(self.skel_nodes)
            dist, ind = tree.query(self.vertices, k=1)

            self._mesh2skel_dict = defaultdict(list)
            for vertex_idx, skel_idx in enumerate(ind):
                self._mesh2skel_dict[skel_idx].append(vertex_idx)

        return self._mesh2skel_dict

    def weighted_graph(self) -> nx.Graph:
        """ Creates a Euclidean distance weighted networkx graph representation of the
        skeleton of this point cloud. The node IDs represent the index in the ``skel_node`` array.

        Returns:
            The weighted skeleton of this point cloud as a networkx graph.
        """
        if self._weighted_graph is None:
            edge_coords = self.skel_nodes[self.skel_edges]
            weights = np.linalg.norm(edge_coords[:, 0] - edge_coords[:, 1], axis=1)

            self._weighted_graph = nx.Graph()

            self._weighted_graph.add_nodes_from(
                [(ix, dict(position=coord)) for ix, coord in
                 enumerate(self.skel_nodes)])

            self._weighted_graph.add_weighted_edges_from(
                [(self.skel_edges[i][0], self.skel_edges[i][1], weights[i]) for
                 i in range(len(weights))])

        return self._weighted_graph

    def simple_graph(self) -> nx.Graph:
        """ Creates a networkx graph representation of the skeleton of this point cloud.
        For a weighted graph see :func:`~weighted_graph`. The node IDs represent the index
        in the ``'skel_node'`` array.

        Returns:
            The skeleton of this point cloud as a networkx graph.
        """
        if self._simple_graph is None:
            self._simple_graph = nx.Graph()

            self._simple_graph.add_nodes_from(
                [(ix, dict(position=coord)) for ix, coord in
                 enumerate(self.skel_nodes)])

            self._simple_graph.add_edges_from(self.skel_edges)

        return self._simple_graph



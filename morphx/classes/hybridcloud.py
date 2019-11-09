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


class HybridCloud(object):
    """
    Class which represents a skeleton in form of a graph structure and a mesh which surrounds this skeleton.
    """

    def __init__(self, skel_nodes: np.ndarray, skel_edges: np.ndarray, vertices: np.ndarray, mesh2skel_dict=None):
        """
        Args:
            skel_nodes: coordinates of the nodes of the skeleton with shape (n, 3).
            skel_edges: edge list with indices of nodes in skel_nodes with shape (n, 2).
            vertices: coordinates of the mesh vertices which surround the skeleton with shape (n, 3).
            mesh2skel_dict: dict structure that maps mesh vertices to skeleton nodes. Keys are skeleton node indices,
                values are lists of mesh vertex indices.
        """
        self._skel_nodes = skel_nodes
        self._skel_edges = skel_edges
        self._vertices = vertices

        self._mesh2skel_dict = None
        if mesh2skel_dict is not None:
            self._mesh2skel_dict = mesh2skel_dict

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

    @property
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

    def graph(self, simple=False) -> nx.Graph:
        """ Creates a Euclidean distance weighted networkx graph representation of the
        skeleton of this point cloud. The node IDs represent the index in the ``skel_node`` array.

        Args:
            simple: Flag for creating a simple graph without the weights

        Returns:
            The skeleton of this point cloud as a (weighted / simple) networkx graph.
        """
        if (self._weighted_graph is None and not simple) or (self._simple_graph is None and simple):
            graph = nx.Graph()

            graph.add_nodes_from(
                [(ix, dict(position=coord)) for ix, coord in
                 enumerate(self.skel_nodes)])

            if simple:
                graph.add_edges_from(self.skel_edges)
                self._simple_graph = graph
            else:
                edge_coords = self.skel_nodes[self.skel_edges]
                weights = np.linalg.norm(edge_coords[:, 0] - edge_coords[:, 1], axis=1)
                graph.add_weighted_edges_from(
                    [(self.skel_edges[i][0], self.skel_edges[i][1], weights[i]) for
                     i in range(len(weights))])
                self._weighted_graph = graph

        if simple:
            return self._simple_graph
        else:
            return self._weighted_graph

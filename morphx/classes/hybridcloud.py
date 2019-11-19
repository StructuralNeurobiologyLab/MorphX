# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
import networkx as nx
import morphx.processing.graphs as graphs
from collections import defaultdict
from scipy.spatial import cKDTree
from morphx.classes.pointcloud import PointCloud


class HybridCloud(PointCloud):
    """
    Class which represents a skeleton in form of a graph structure and a mesh which surrounds this skeleton.
    """

    def __init__(self, nodes: np.ndarray, edges: np.ndarray, vertices: np.ndarray, vert2skel=None, labels=None):
        """
        Args:
            nodes: coordinates of the nodes of the skeleton with shape (n, 3).
            edges: edge list with indices of nodes in skel_nodes with shape (n, 2).
            vertices: coordinates of the mesh vertices which surround the skeleton with shape (n, 3).
            vert2skel: dict structure that maps mesh vertices to skeleton nodes. Keys are skeleton node indices,
                values are lists of mesh vertex indices.
            labels: vertex label array (integer number representing respective classe) with same dimensions as
                vertices.
        """
        super().__init__(vertices, labels=labels)

        self._nodes = nodes
        self._edges = edges

        self._vert2skel = None
        if vert2skel is not None:
            self._vert2skel = vert2skel

        self._weighted_graph = None
        self._simple_graph = None
        self._traverser = None

    @property
    def nodes(self) -> np.ndarray:
        """ Coordinates of the nodes of the skeleton.
        """
        return self._nodes

    @property
    def edges(self) -> np.ndarray:
        """ Edge list with indices of nodes in skel_nodes as np.ndarray with shape (n, 2).
        """
        return self._edges

    @property
    def vert2skel(self) -> defaultdict:
        """ Creates python defaultdict with indices of skel_nodes as keys and lists of vertex
        indices which have their key node as nearest skeleton node.

        Returns:
            Python defaultdict with mapping information
        """
        if self._vert2skel is None:
            tree = cKDTree(self.nodes)
            dist, ind = tree.query(self.vertices, k=1)

            self._vert2skel = defaultdict(list)
            for vertex_idx, skel_idx in enumerate(ind):
                self._vert2skel[skel_idx].append(vertex_idx)

        return self._vert2skel

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
                 enumerate(self.nodes)])

            if simple:
                graph.add_edges_from(self.edges)
                self._simple_graph = graph
            else:
                edge_coords = self.nodes[self.edges]
                weights = np.linalg.norm(edge_coords[:, 0] - edge_coords[:, 1], axis=1)
                graph.add_weighted_edges_from(
                    [(self.edges[i][0], self.edges[i][1], weights[i]) for
                     i in range(len(weights))])
                self._weighted_graph = graph

        if simple:
            return self._simple_graph
        else:
            return self._weighted_graph

    def traverser(self, method='global_bfs', min_dist=0, source=-1) -> np.ndarray:
        """ Creates or returns an array of node indices which can be used as the order in which the hybrid
        should be traversed. With method = 'global_bfs', this method calculates the global BFS for the weighted
        graph of this hybrid object.

        Args:
            method: The method with which the order array should be created.
                'global_bfs' for global BFS with minimum distance.
            min_dist: The minimum distance between points in the result of the BFS.
            source: The starting point of the BFS.

        Returns:
              Array with resulting nodes from a BFS where all nodes have a minimum distance of min_dist to each other.
        """
        if self._traverser is None:
            if method == 'global_bfs':
                self._traverser = graphs.global_bfs_dist(self.graph(), min_dist, source)

        return self._traverser

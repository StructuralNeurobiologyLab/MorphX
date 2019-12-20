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
    """ Class which represents a skeleton in form of a graph structure and vertices which surround this skeleton and
        which represent the actual shape of the object. """

    def __init__(self,
                 nodes: np.ndarray,
                 edges: np.ndarray,
                 vertices: np.ndarray,
                 vert2skel: defaultdict = None,
                 labels: np.ndarray = None,
                 encoding: dict = None):
        """
        Args:
            nodes: Coordinates of the nodes of the skeleton with shape (n, 3).
            edges: Edge list with indices of nodes in skel_nodes with shape (n, 2).
            vertices: Coordinates of the mesh vertices which surround the skeleton with shape (n, 3).
            vert2skel: Dict structure that maps mesh vertices to skeleton nodes. Keys are skeleton node indices,
                values are lists of mesh vertex indices.
            labels: Vertex label array (integer number representing respective classe) with same dimensions as
                vertices.
            encoding: Dict with unique labels as keys and description string for respective label as value.
        """
        super().__init__(vertices, labels=labels, encoding=encoding)

        if nodes.shape[1] != 3:
            raise ValueError("Nodes must have shape (N, 3).")
        self._nodes = nodes

        if edges.max() > len(nodes):
            raise ValueError("Edge list cannot contain indices which exceed the size of the node array.")
        self._edges = edges

        self._vert2skel = None
        if vert2skel is not None:
            self._vert2skel = vert2skel

        self._weighted_graph = None
        self._simple_graph = None
        self._traverser = None

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes

    def set_nodes(self, nodes) -> None:
        if nodes.shape != self._nodes.shape:
            raise ValueError("Shape of nodes must not change as edges would loose their reference.")
        self._nodes = nodes

    @property
    def edges(self) -> np.ndarray:
        return self._edges

    @property
    # think about renaming it into verts2nodes
    def vert2skel(self) -> dict:
        """ Creates python defaultdict with indices of skel_nodes as keys and lists of vertex
        indices which have their key node as nearest skeleton node.

        Returns:
            Python defaultdict with mapping information
        """
        if self._vert2skel is None:
            tree = cKDTree(self.nodes)
            dist, ind = tree.query(self.vertices, k=1)

            self._vert2skel = {ix: [] for ix in range(len(self.nodes))}
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
        should be traversed. With ``method = 'global_bfs'``, this method calculates the global BFS for the weighted
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
                self._traverser = graphs.global_bfs_dist(self.graph(), min_dist, source=source)

        return self._traverser

    def filter_traverser(self):
        """ Removes all nodes from `:py:func:~traverser` which have no vertices to which they are the nearest node.
        Used for making sure that label filtered clouds can be traversed without processing overhead for nodes without
        vertices.

        Returns:
            Filtered traverser array.
        """
        # TODO: This could remove potentially useful nodes => Improve and include better criteria for removal
        f_traverser = []
        mapping = self.vert2skel
        for node in self.traverser():
            # get only those nodes which are the nearest neighbors to some vertices
            if len(mapping[node]) != 0:
                f_traverser.append(node)
        f_traverser = np.array(f_traverser)
        self._traverser = f_traverser

        return f_traverser

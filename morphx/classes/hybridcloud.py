# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import warnings
import numpy as np
import networkx as nx
from typing import Optional
from scipy.spatial import cKDTree
from morphx.classes.pointcloud import PointCloud
from scipy.spatial.transform import Rotation as Rot


class HybridCloud(PointCloud):
    """ Class which represents a skeleton in form of a graph structure and vertices which surround this skeleton and
        which represent the actual shape of the object. """

    def __init__(self,
                 nodes: np.ndarray = None,
                 edges: np.ndarray = None,
                 verts2node: dict = None,
                 node_labels: np.ndarray = None,
                 pred_node_labels: np.ndarray = None,
                 *args, **kwargs):
        """
        Args:
            nodes: Coordinates of the nodes of the skeleton with shape (n, 3).
            edges: Edge list with indices of nodes in skel_nodes with shape (n, 2).
            verts2node: Dict structure that maps mesh vertices to skeleton nodes. Keys are skeleton node indices,
                values are lists of mesh vertex indices.
            node_labels: Node label array (ith label corresponds to ith node) with same dimenstions as nodes.
        """
        super().__init__(*args, **kwargs)
        if nodes is not None and nodes.shape[1] != 3:
            raise ValueError("Nodes must have shape (N, 3).")
        self._nodes = nodes

        if nodes is None:
            self._edges = None
            self._node_labels = None
        else:
            if edges is not None and edges.max() > len(nodes):
                raise ValueError("Edge list cannot contain indices which exceed the size of the node array.")
            self._edges = edges.astype(int)

            self._node_labels = None
            if node_labels is not None:
                if len(node_labels) != len(nodes):
                    raise ValueError("Node label array must have same length as nodes array.")
                self._node_labels = node_labels.reshape(len(node_labels), 1)

        self._pred_node_labels = pred_node_labels
        self._verts2node = None
        if verts2node is not None:
            self._verts2node = verts2node

        self._weighted_graph = None
        self._simple_graph = None
        self._base_points = None

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes

    @property
    def edges(self) -> np.ndarray:
        return self._edges

    @property
    def verts2node(self) -> dict:
        """ Creates python dict with indices of skel_nodes as keys and lists of vertex
        indices which have their key node as nearest skeleton node.

        Returns:
            Dict with mapping information
        """
        if self._verts2node is None:
            tree = cKDTree(self.nodes)
            dist, ind = tree.query(self.vertices, k=1)

            self._verts2node = {ix: [] for ix in range(len(self.nodes))}
            for vertex_idx, skel_idx in enumerate(ind):
                self._verts2node[skel_idx].append(vertex_idx)
        return self._verts2node

    @property
    def node_labels(self):
        if self._node_labels is None:
            self._node_labels = self.vertl2nodel(pred=False)
        return self._node_labels

    @property
    def pred_node_labels(self):
        if self._pred_node_labels is None:
            self._pred_node_labels = self.vertl2nodel(pred=True)
        return self._pred_node_labels

    # -------------------------------------- SETTERS ------------------------------------------- #

    def set_node_labels(self, node_labels: np.ndarray):
        if len(node_labels) != len(self._nodes):
            raise ValueError('Length of node_labels must comply with length of nodes.')
        self._node_labels = node_labels
        
    def set_verts2node(self, verts2node: dict):
        if len(verts2node) != len(self._nodes):
            raise ValueError('Length of verts2nodes must comply with length of nodes.')
        self._verts2node = verts2node

    # -------------------------------------- HYBRID BASICS ------------------------------------------- #

    def clean_node_labels(self, neighbor_num: int = 2):
        """ For each node, this method performs a majority vote on the labels of neighbor_num neighbors which were
        extracted with a limited bfs. The most frequent label is used for the current node. This method can be used as
        a sliding window filter for removing single or small groups of wrong labels on the skeleton.

        Args:
            neighbor_num: Number of neighbors which limits the radius of the bfs.
        """
        from morphx.processing import graphs
        if self.node_labels is None:
            return
        new_labels = np.zeros((len(self.node_labels), 1))
        graph = self.graph(simple=True)

        # for each node extract neighbor_num neighbors with a bfs and take the most frequent label as new node label
        for ix in range(len(self.nodes)):
            local_bfs = graphs.bfs_num(graph, ix, neighbor_num)
            labels = self.node_labels[local_bfs.astype(int)]
            u_labels, counts = np.unique(labels, return_counts=True)
            new_labels[ix] = u_labels[np.argmax(counts)]
        self._node_labels = new_labels

    def vertl2nodel(self, pred: bool = True) -> Optional[np.ndarray]:
        """ Uses verts2node to transfer vertex labels onto the skeleton. For each node, a majority vote on the labels of
         the corresponding vertices is performed and the most frequent label is transferred to the node.
         Returns:
             None if there are no vertex labels or a np.ndarray with the node labels (ith label corresponds to ith node)
         """
        from morphx.processing import hybrids
        if pred:
            vertl = self._pred_labels
        else:
            vertl = self._labels
        if vertl is None:
            return None
        else:
            nodel = np.zeros((len(self._nodes), 1), dtype=int)
            nodel[:] = -1
            # extract vertices corresponding to each node and take the majority label as the label for that node
            for ix in range(len(self._nodes)):
                verts_idcs = self.verts2node[ix]
                # nodes with no corresponding vertices have label -1
                if len(verts_idcs) != 0:
                    labels = vertl[verts_idcs]
                    # remove unpredicted nodes (no action if pred == False)
                    labels = labels[labels != -1]
                    if len(labels) == 0:
                        continue
                    u_labels, counts = np.unique(labels, return_counts=True)
                    # take first label if there are multiple majorities
                    nodel[ix] = u_labels[np.argmax(counts)]
            # nodes without label (still == -1) get label from nearest node with label
            mapping = np.arange(len(self._nodes))
            if np.all(nodel == -1):
                warnings.warn("All node labels have label -1. Label mapping was not possible.")
                return nodel
            for ix in range(len(self._nodes)):
                if nodel[ix] == -1:
                    mapping[ix] = hybrids.label_search(self, nodel, ix)
            nodel = nodel[mapping]
        return nodel

    def nodel2vertl(self):
        """ Uses the verts2node dict to map labels from nodes onto vertices. """
        if self._node_labels is None:
            return
        else:
            for ix in range(len(self._nodes)):
                verts_idcs = self.verts2node[ix]
                self._pred_labels[verts_idcs] = self._node_labels[ix]

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

    def base_points(self, density_mode: bool = True, threshold: int = 0, source: int = -1) -> np.ndarray:
        """ Creates base points on the graph of the hybrid. These points can be used to extract local
            contexts.

        Args:
            density_mode: flag for (de)activating density mode.
            threshold: the minimum distance between points in the result of the BFS.
            source: the starting point of the BFS.

        Returns:
              Array with resulting nodes from a BFS where all nodes have a minimum distance of min_dist to each other.
        """
        from morphx.processing import graphs, objects
        if self._base_points is None:
            if density_mode:
                self._base_points = objects.bfs_base_points_density(self, threshold, source=source)
            else:
                self._base_points = graphs.bfs_base_points_euclid(self.graph(), threshold, source=source)
        return self._base_points

    def filter_traverser(self):
        """ Removes all nodes from `:py:func:~traverser` which have no vertices to which they are the nearest node.
        Used for making sure that label filtered clouds can be traversed without processing overhead for nodes without
        vertices.

        Returns:
            Filtered traverser array.
        """
        # TODO: This could remove potentially useful nodes => Improve and include better criteria for removal
        f_traverser = []
        mapping = self.verts2node
        for node in self.base_points():
            # get only those nodes which are the nearest neighbors to some vertices
            if len(mapping[node]) != 0:
                f_traverser.append(node)
        f_traverser = np.array(f_traverser)
        self._base_points = f_traverser
        return f_traverser

    # -------------------------------------- TRANSFORMATIONS ------------------------------------------- #

    def scale(self, factor: int):
        """ If factor < 0 vertices and nodes are divided by the factor. If factor > 0 vertices and nodes are
            multiplied by the factor. If factor == 0 nothing happens. """
        if factor == 0:
            return
        elif factor < 0:
            self._vertices = self._vertices / -factor
            self._nodes = self._nodes / -factor
        else:
            self._vertices = self._vertices * factor
            self._nodes = self.nodes * factor

    def rotate_randomly(self, angle_range: tuple = (-180, 180)):
        """ Randomly rotates vertices and nodes by performing an Euler rotation. The three angles are choosen randomly
            from the given angle_range. """
        # switch limits if lower limit is larger
        if angle_range[0] > angle_range[1]:
            angle_range = (angle_range[1], angle_range[0])

        angles = np.random.uniform(angle_range[0], angle_range[1], (1, 3))[0]
        r = Rot.from_euler('xyz', angles, degrees=True)
        if len(self._vertices) > 0:
            self._vertices = r.apply(self._vertices)
        if len(self._nodes) > 0:
            self._nodes = r.apply(self._nodes)

    def move(self, vector: np.ndarray):
        """ Moves vertices and nodes by adding the given vector """
        self._vertices = self._vertices + vector
        self._nodes = self._nodes + vector

    # -------------------------------------- HYBRID I/O ------------------------------------------- #

    def get_attr_dict(self):
        attr_dict = {'nodes': self._nodes, 'edges': self._edges, 'verts2node': self.verts2node,
                     'pred_node_labels': self._pred_node_labels, 'node_labels': self._node_labels}
        attr_dict.update(super().get_attr_dict())
        return attr_dict

# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import warnings
import numpy as np
import logging
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
        if nodes is None:
            nodes = np.zeros((0, 3))
        if nodes.shape[1] != 3:
            raise ValueError("Nodes must have shape (N, 3).")
        self._nodes = nodes

        if edges is None:
            edges = np.zeros((0, 2))
        if len(edges) != 0 and edges.max() > len(nodes):
            raise ValueError("Edge list cannot contain indices which exceed the size of the node array.")
        self._edges = edges.astype(int)

        if node_labels is None:
            node_labels = np.zeros((0, 1))
        if len(node_labels) != 0 and len(node_labels) != len(nodes):
            raise ValueError("Node label array must have same length as nodes array.")
        self._node_labels = node_labels.reshape(len(node_labels), 1).astype(int)

        if pred_node_labels is None:
            pred_node_labels = np.zeros((0, 1))
        if len(pred_node_labels) != 0 and len(pred_node_labels) != len(nodes):
            raise ValueError("Predicted node label array must have same length as nodes array")
        self._pred_node_labels = pred_node_labels.reshape(len(pred_node_labels), 1).astype(int)

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
    def verts2node(self) -> Optional[dict]:
        """ Creates python dict with indices of skel_nodes as keys and lists of vertex
        indices which have their key node as nearest skeleton node.

        Returns:
            Dict with mapping information
        """
        if len(self._nodes) == 0:
            return None
        if self._verts2node is None:
            tree = cKDTree(self.nodes)
            dist, ind = tree.query(self.vertices, k=1)

            self._verts2node = {ix: [] for ix in range(len(self.nodes))}
            for vertex_idx, skel_idx in enumerate(ind):
                self._verts2node[skel_idx].append(vertex_idx)
        return self._verts2node

    @property
    def node_labels(self):
        if len(self._node_labels) == 0:
            self._node_labels = self.vertl2nodel(pred=False)
        return self._node_labels

    @property
    def pred_node_labels(self):
        if len(self._pred_node_labels) == 0:
            self._pred_node_labels = self.vertl2nodel(pred=True)
        return self._pred_node_labels

    # -------------------------------------- SETTERS ------------------------------------------- #

    def set_node_labels(self, node_labels: np.ndarray):
        if len(node_labels) != len(self._nodes):
            raise ValueError('Length of node_labels must comply with length of nodes.')
        self._node_labels = node_labels

    def set_pred_node_labels(self, pred_node_labels: np.ndarray):
        if len(pred_node_labels) != len(self._nodes):
            raise ValueError('Length of node_labels must comply with length of nodes.')
        self._pred_node_labels = pred_node_labels
        
    def set_verts2node(self, verts2node: dict):
        if verts2node is None:
            self._verts2node = None
            return
        if len(verts2node) != len(self._nodes):
            raise ValueError('Length of verts2nodes must comply with length of nodes.')
        self._verts2node = verts2node

    # -------------------------------------- HYBRID BASICS ------------------------------------------- #

    def clean_node_labels(self, neighbor_num: int = 20):
        """ For each node, this method performs a majority vote on the labels of neighbor_num neighbors which were
        extracted with a limited bfs. The most frequent label is used for the current node. This method can be used as
        a sliding window filter for removing single or small groups of wrong labels on the skeleton.

        Args:
            neighbor_num: Number of neighbors which limits the radius of the bfs.
        """
        from morphx.processing import graphs
        new_labels = np.zeros((len(self.pred_node_labels), 1))
        graph = self.graph(simple=True)

        # for each node extract neighbor_num neighbors with a bfs and take the most frequent label as new node label
        for ix in range(len(self.nodes)):
            local_bfs = graphs.bfs_num(graph, ix, neighbor_num)
            labels = self.pred_node_labels[local_bfs.astype(int)]
            u_labels, counts = np.unique(labels, return_counts=True)
            new_labels[ix] = u_labels[np.argmax(counts)]
        self._pred_node_labels = new_labels

    def vertl2nodel(self, pred: bool = True) -> np.ndarray:
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
        if len(vertl) == 0:
            return np.zeros((0, 1))
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

    def prednodel2predvertl(self):
        """ Uses the verts2node dict to map labels from nodes onto vertices. """
        if len(self._pred_node_labels) == 0:
            return
        if len(self._pred_labels) == 0:
            self._pred_labels = np.ones((len(self._vertices), 1)) * -1
        else:
            for ix in range(len(self._nodes)):
                verts_idcs = self.verts2node[ix]
                self._pred_labels[verts_idcs] = self._pred_node_labels[ix]

    def nodel2vertl(self):
        if len(self._node_labels) == 0:
            return
        if len(self._labels) == 0:
            self._labels = np.ones((len(self._vertices), 1)) * -1
        for ix in range(len(self._nodes)):
            verts_idcs = self.verts2node[ix]
            self._labels[verts_idcs] = self._node_labels[ix]

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

    def base_points(self, threshold: int = 0, source: int = -1) -> np.ndarray:
        """ Creates base points on the graph of the hybrid. These points can be used to extract local
            contexts.

        Args:
            threshold: the minimum distance between points in the result of the BFS.
            source: the starting point of the BFS.

        Returns:
              Array with resulting nodes from a BFS where all nodes have a minimum distance of min_dist to each other.
        """
        from morphx.processing import graphs
        if self._base_points is None:
            self._base_points = graphs.bfs_base_points_euclid(self.graph(), threshold, source=source)
        return self._base_points

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

    def rotate_randomly(self, angle_range: tuple = (-180, 180), random_flip: bool = False):
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
        if random_flip:
            flip_axes = (-1)**np.random.randint(0, 2, self._vertices.shape[1])
            self._vertices *= flip_axes
            self._nodes *= flip_axes

    def move(self, vector: np.ndarray):
        """ Moves vertices and nodes by adding the given vector """
        self._vertices = self._vertices + vector
        self._nodes = self._nodes + vector

    def add_noise(self, limits: tuple = (-1, 1), distr: str = 'uniform',
                  include_nodes=True):
        """
        Apply additive noise (drawn from `distr` and scaled by `distr_scale`) to vertices.

        Args:
            limits: Range of the noise values. Tuple is used as lower and upper bounds for ``distr='uniform'``
                or only the entry at index 1 is used as standard deviation if ``distr='normal'``. Note that the
                s.d. used to generate the vertex noise (i.i.d) is fixed by drawing a global value from the given normal
                distribution. This will lead to different noise levels within the given s.d. range (limits[1]).
            distr: Noise distribution, currently available: 'uniform' and 'Gaussian'.
            include_nodes: If True, noise will be applied to nodes.

        Returns:

        """
        if distr.lower() == 'normal':
            if abs(limits[0]) != abs(limits[1]):
                logging.warning(f'Lower ({limits[0]}) and upper ({limits[1]}) limits differ but chosen '
                                f'noise source was set to "normal". Using upper limit to scale standard '
                                f'normal values.')
            fixed_sd = np.random.standard_normal(1) * limits[1]
            variation = np.random.standard_normal(self._vertices.shape) * fixed_sd
            variation_nodes = np.random.standard_normal(self._nodes.shape) * fixed_sd
        elif distr.lower() == 'uniform':
            # switch limits if lower limit is larger
            if limits[0] > limits[1]:
                limits = (limits[1], limits[0])
            # do nothing if limits are the same
            if limits[0] == limits[1]:
                return
            variation = np.random.random(self._vertices.shape) * (limits[1] - limits[0]) + limits[0]
            variation_nodes = np.random.random(self._nodes.shape) * (limits[1] - limits[0]) + limits[0]
        else:
            raise ValueError(f'Given value "{distr}" for noise distribution not available.')
        self._vertices = self._vertices + variation
        if include_nodes:
            self._nodes += variation_nodes

    # mult_noise of PointCloud generalizes to HybridCloud

    def shear(self, limits: tuple = (-1, 1)):
        """
        Shears vertices and nodes by applying a transformation matrix
        [[1, s_xy, s_xz], [s_yx, 1, s_yz], [s_zx, s_zy, 1]], where the factors
        s_ij are drawn from a uniform distribution.

        Args:
            limits: Range of the interval for the factors s_ij. Tuple defines lower
                and upper bound of the uniform distribution.
        """
        transform = np.random.random((3, 3)) * (limits[1] - limits[0]) + limits[0]
        np.fill_diagonal(transform, 1)
        self._vertices.dot(transform)
        self._nodes.dot(transform)

    # -------------------------------------- HYBRID I/O ------------------------------------------- #

    def get_attr_dict(self):
        attr_dict = {'nodes': self._nodes, 'edges': self._edges, 'verts2node': self.verts2node,
                     'pred_node_labels': self._pred_node_labels, 'node_labels': self._node_labels}
        attr_dict.update(super().get_attr_dict())
        return attr_dict

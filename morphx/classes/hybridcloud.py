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
from typing import Optional, List, Dict, Tuple
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
            node_labels: Node label array (ith label corresponds to ith node) with same dimensions as nodes.
        """
        super().__init__(*args, **kwargs)
        if nodes is None:
            nodes = np.zeros((0, 3))
        if nodes.shape[1] != 3:
            raise ValueError("Nodes must have shape (N, 3).")
        self._nodes = np.array(nodes, dtype=np.float32)  # trigger copy

        if edges is None:
            edges = np.zeros((0, 2))
        if len(edges) != 0 and edges.max() > len(nodes):
            raise ValueError("Edge list cannot contain indices which exceed the size of the node array.")
        self._edges = np.array(edges).astype(int)

        if node_labels is None:
            node_labels = np.zeros((0, 1))
        if len(node_labels) != 0 and len(node_labels) != len(nodes):
            raise ValueError("Node label array must have same length as nodes array.")
        self._node_labels = np.array(node_labels.reshape(len(node_labels), 1)).astype(int)

        if pred_node_labels is None:
            pred_node_labels = np.zeros((0, 1))
        if len(pred_node_labels) != 0 and len(pred_node_labels) != len(nodes):
            raise ValueError("Predicted node label array must have same length as nodes array")
        self._pred_node_labels = np.array(pred_node_labels.reshape(len(pred_node_labels), 1)).astype(int)

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
        """ Generates node labels from vertex labels if node label array is empty. """
        if len(self._node_labels) == 0:
            self._node_labels = self.vertl2nodel(pred=False)
        return self._node_labels

    @property
    def pred_node_labels(self):
        """ Generates node predictions from vertex predicitons if node prediction array is empty. """
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

    def node_sliding_window_bfs(self, neighbor_num: int = 20, predictions: bool = True):
        """ For each node, this method performs a majority vote on the labels of neighbor_num neighbors which were
        extracted with a limited bfs. The most frequent label is used for the current node. This method can be used as
        a sliding window filter for removing single or small groups of wrong labels on the skeleton.

        Args:
            neighbor_num: Number of neighbors which limits the radius of the bfs.
            predictions: Flag for applying sliding window to node predictions or to existing node labels.
        """
        from morphx.processing import graphs
        if predictions:
            node_labels = self.pred_node_labels
        else:
            node_labels = self.node_labels
        new_labels = np.zeros((len(node_labels), 1))
        graph = self.graph(simple=True)
        # for each node extract neighbor_num neighbors with a bfs and take the most frequent label as new node label
        for ix in range(len(self.nodes)):
            local_bfs = graphs.bfs_num(graph, ix, neighbor_num)
            labels = node_labels[local_bfs.astype(int)]
            u_labels, counts = np.unique(labels, return_counts=True)
            new_labels[ix] = u_labels[np.argmax(counts)]
        if predictions:
            self._pred_node_labels = new_labels
        else:
            self._node_labels = new_labels

    def vertl2nodel(self, pred: bool = True, propagate: bool = False) -> np.ndarray:
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
            if propagate:
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
                if len(self._nodes) == 0:
                    self._weighted_graph = graph
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

    def remove_nodes(self, labels: List[int], threshold: int = 20) -> Dict:
        """ Removes all nodes with labels present in the given labels list. This method also updates the
            verts2node mapping accordingly. Vertices which belong to removed nodes don't appear in the
            verts2node mapping.

        Args:
            labels: List of labels to indicate which nodes should get removed.
            threshold: Connected components where number of nodes is below this threshold get removed.
        """
        if labels is None or len(labels) == 0:
            return {}
        _ = self.verts2node
        # generate node labels by mapping vertex labels to nodes
        if len(self.node_labels) == 0:
            return {}
        mask = np.isin(self._node_labels, labels).reshape(-1)
        rnodes = np.arange(len(self._nodes))[mask]
        graph = self.graph()
        # change graph
        for rnode in rnodes:
            graph.remove_node(rnode)
        # filter small outliers
        for cc in list(nx.connected_components(graph)):
            if len(cc) < threshold:
                for node in cc:
                    graph.remove_node(node)
        # change corresponding arrays
        self._pred_node_labels = np.zeros((0, 1))
        if len(graph.nodes) == 0:
            # no nodes are left
            self._nodes = np.zeros((0, 3))
            self._node_labels = np.zeros((0, 1))
            self._verts2node = {}
            self._edges = np.zeros((0, 2))
            self._simple_graph = None
            self._weighted_graph = None
            return {}
        self._nodes = self._nodes[np.array(graph.nodes)]
        self._node_labels = self._node_labels[np.array(graph.nodes)]
        # relabel nodes to consecutive numbers and update edges of HybridCloud
        mapping = {i: x for x, i in enumerate(graph.nodes)}
        graph = nx.relabel_nodes(graph, mapping)
        self._edges = np.array(graph.edges)
        self._simple_graph = None
        self._weighted_graph = None
        # Update verts2node array
        new_verts2node = {}
        for key in self.verts2node:
            if key in mapping:
                new_verts2node[mapping[key]] = self.verts2node[key]
                # relabel remaining vertices
                mask = np.isin(self._labels[self.verts2node[key]], labels).reshape(-1)
                if np.any(mask):
                    idcs = np.array(self.verts2node[key])[mask]
                    self._labels[idcs] = self._node_labels[mapping[key]]
        self._verts2node = new_verts2node
        return mapping

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

    def map_labels(self, mappings: List[Tuple[int, int]]):
        """ In-place method for changing labels of vertices and nodes. Encoding gets updated.

        Args:
            mappings: List of tuples with original labels and target labels. E.g. [(1, 2), (3, 2)] means that
              the labels 1 and 3 will get replaced by 2.
        """
        for mapping in mappings:
            self._labels[self._labels == mapping[0]] = mapping[1]
            self._node_labels[self._node_labels == mapping[0]] = mapping[1]
            if self._encoding is not None and mapping[0] in self._encoding:
                self._encoding.pop(mapping[0], None)

    # -------------------------------------- TRANSFORMATIONS ------------------------------------------- #

    def scale(self, factor: Optional[int]):
        """ If factor < 0 vertices and nodes are divided by the factor. If factor > 0 vertices and nodes are
            multiplied by the factor. If factor == 0 nothing happens. """
        if factor is None:
            factor = -np.max(np.abs(self._vertices))
        if np.any(factor == 0):
            return
        if np.isscalar(factor):
            factor = np.array([factor] * 3)
        elif type(factor) is not np.ndarray:
            factor = np.array(factor)
        if np.any(factor < 0):
            self._vertices[..., factor < 0] = self._vertices[..., factor < 0] / -factor[factor < 0]
            self._nodes[..., factor < 0] = self._nodes[..., factor < 0] / -factor[factor < 0]
        self._vertices[..., factor > 0] = self._vertices[..., factor > 0] * factor[factor > 0]
        self._nodes[..., factor > 0] = self._nodes[..., factor > 0] * factor[factor > 0]

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

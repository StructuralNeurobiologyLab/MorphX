# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from typing import Dict, Optional
from morphx.classes.pointcloud import PointCloud
from morphx.classes.hybridcloud import HybridCloud
from morphx.processing import ensembles


class CloudEnsemble(object):
    """
    Class which represents a collection of PointCloud objects.
    """

    def __init__(self, clouds: Dict[str, PointCloud], hybrid: Optional[HybridCloud] = None):
        """
        Args:
            clouds: Dict with cloud names as keys and PointCloud objects as Values. Objects like HybridClouds in this
                dict get treated as sole PointClouds.
            hybrid: The HypridCloud on which all graph and extraction algorithms are performed for this ensemble.
        """
        self._clouds = clouds
        self._hc = hybrid
        self._verts2node = None

    @property
    def clouds(self):
        return self._clouds

    @property
    def nodes(self) -> np.ndarray:
        if self._hc is None:
            return np.empty((0, 3))
        return self._hc.nodes

    @property
    def edges(self) -> np.ndarray:
        if self._hc is None:
            return np.empty((0, 2))
        return self._hc.edges

    @property
    def hc(self):
        return self._hc

    @property
    def verts2node(self) -> dict:
        """ Creates python dict with indices of skel_nodes as keys and lists of vertex
        indices which have their key node as nearest skeleton node.

        Returns:
            Dict with mapping information
        """
        if self._verts2node is None:
            self._verts2node = {}
            merged = ensembles.ensemble2pointcloud(self)
            if isinstance(merged, HybridCloud):
                if len(merged.nodes) > 0:
                    tree = cKDTree(merged.nodes)
                    dist, ind = tree.query(merged.vertices, k=1)

                    self._verts2node = {ix: [] for ix in range(len(merged.nodes))}
                    for vertex_idx, skel_idx in enumerate(ind):
                        self._verts2node[skel_idx].append(vertex_idx)
        return self._verts2node

    def base_points(self, method='global_bfs', min_dist=0, source=-1) -> np.ndarray:
        if self._hc is None:
            np.zeros(0)
        else:
            return self._hc.base_points(method=method, min_dist=min_dist, source=source)

    def graph(self, simple=False) -> nx.Graph:
        if self._hc is None:
            return nx.Graph()
        else:
            return self._hc.graph(simple=simple)

    def get_cloud(self, cloud_name: str) -> Optional[PointCloud]:
        try:
            return self._clouds[cloud_name]
        except ValueError:
            return None

    def add_cloud(self, cloud: PointCloud, cloud_name: str):
        self._clouds[cloud_name] = cloud
        self._reset_ensemble()

    def change_hybrid(self, hybrid: HybridCloud):
        self._hc = hybrid
        self._reset_ensemble()

    def _reset_ensemble(self):
        self._verts2node = None

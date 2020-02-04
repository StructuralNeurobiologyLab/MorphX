# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
import networkx as nx
from typing import Dict, Optional
from morphx.classes.pointcloud import PointCloud
from morphx.classes.hybridcloud import HybridCloud


class CloudEnsemble(object):
    """
    Class which represents a collection of PointCloud objects.
    """

    def __init__(self, clouds: Dict[str, PointCloud]):
        """
        Args:
            clouds: Dict with cloud names as keys and PointCloud objects as Values.
        """
        self._clouds = clouds
        if len(self._clouds) == 0:
            raise ValueError("CloudEnsemble must contain at least one cloud.")
        self._hc = None
        for cloud in self._clouds:
            if isinstance(cloud, HybridCloud):
                if self._hc is None:
                    self._hc = cloud
                else:
                    raise ValueError("CloudEnsemble can only contain one HybridCloud.")

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

    def graph(self, simple=False) -> Optional[nx.Graph]:
        if self._hc is None:
            return None
        else:
            return self._hc.graph(simple=simple)

    def get_cloud(self, cloud_name: str) -> Optional[PointCloud]:
        try:
            return self._clouds[cloud_name]
        except ValueError:
            return None

    def set_cloud(self, cloud: PointCloud, cloud_name: str):
        if isinstance(cloud, HybridCloud) and self._hc is not None:
            raise ValueError("CloudEnsemble can only contain one HybridCloud.")
        elif isinstance(cloud, HybridCloud):
            self._hc = cloud
        self._clouds[cloud_name] = cloud

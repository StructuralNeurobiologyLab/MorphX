# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import time
import pickle
import numpy as np
import networkx as nx
from typing import Optional, List, Dict
from morphx.classes.pointcloud import PointCloud
from morphx.classes.hybridcloud import HybridCloud


class CloudEnsemble(object):
    """
    Class which represents a collection of PointCloud objects.
    """

    def __init__(self, clouds: Dict[str, PointCloud], hybrid: Optional[HybridCloud] = None, no_pred: List[str] = None,
                 predictions: Optional[dict] = None, verts2node: dict = None):
        """
        Args:
            clouds: Dict with cloud names as keys and PointCloud objects as Values. Objects like HybridClouds in this
                dict get treated as sole PointClouds.
            hybrid: The HypridCloud on which all graph and extraction algorithms are performed for this ensemble.
            no_pred: List of names of objects which should not be processed in model prediction or mapping.
            predictions: Dict with vertex indices as keys and prediction lists as values. E.g. if vertex with index 1
                got the labels 2, 3, 4 as predictions, it would be {1: [2, 3, 4]}. Refers to the flattened ensemble.
        """
        self._clouds = clouds
        self._hc = hybrid
        self._flattened = None
        self._verts2node = verts2node
        self._predictions = predictions

        if no_pred is None:
            self._no_pred = []
        else:
            self._no_pred = no_pred

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
    def no_pred(self) -> List[str]:
        return self._no_pred

    @property
    def encoding(self) -> dict:
        return self._hc.encoding

    @property
    def hc(self):
        return self._hc

    @property
    def flattened(self):
        if self._flattened is None:
            # avoids cyclic import
            from morphx.processing import ensembles
            self._flattened = ensembles.ensemble2pointcloud(self)
            self._flattened.add_no_pred(self._no_pred)
            if self._predictions is not None:
                self._flattened.set_predictions(self._predictions)
            if self._verts2node is not None and isinstance(self._flattened, HybridCloud):
                self._flattened.set_verts2node(self._verts2node)
        return self._flattened

    @property
    def predictions(self) -> dict:
        if self._predictions is None:
            self._predictions = self.flattened.predictions
        return self._predictions

    @property
    def verts2node(self) -> dict:
        return self.flattened.verts2node

    def base_points(self, density_mode: bool = True, threshold: int = 0, source: int = -1) -> np.ndarray:
        if isinstance(self.flattened, HybridCloud):
            return self.flattened.base_points(density_mode=density_mode, threshold=threshold, source=source)
        else:
            return np.ndarray([])

    def graph(self, simple=False) -> nx.Graph:
        if self._hc is None:
            return nx.Graph()
        else:
            return self._hc.graph(simple=simple)

    def get_cloud(self, cloud_name: str) -> Optional[PointCloud]:
        if cloud_name == 'hc':
            return self._hc
        try:
            return self._clouds[cloud_name]
        except KeyError:
            return None

    def add_cloud(self, cloud: PointCloud, cloud_name: str):
        self._clouds[cloud_name] = cloud
        self._reset_ensemble()

    def remove_cloud(self, cloud_name: str):
        try:
            del self._clouds[cloud_name]
        except KeyError:
            return

    def change_hybrid(self, hybrid: HybridCloud):
        self._hc = hybrid
        self._reset_ensemble()

    def _reset_ensemble(self):
        self._verts2node = None

    def add_no_pred(self, obj_names: List[str]):
        for name in obj_names:
            if name not in self._no_pred:
                self._no_pred.append(name)

    # -------------------------------------- PREDICTION HANDLING ------------------------------------------- #

    def preds2labels(self, mv: bool = True):
        """ Transfers predictions (gathered for the flattened CloudEnsemble) to the labels of each object in the
            CloudEnsemble. """
        self.flattened.preds2labels(mv)
        hc_bounds = self.flattened.obj_bounds['hybrid']
        self.hc.set_labels(self.flattened.labels[hc_bounds[0]:hc_bounds[1]])
        for key in self.clouds:
            cloud_bounds = self.flattened.obj_bounds[key]
            self._clouds[key].set_labels(self.flattened.labels[cloud_bounds[0]:cloud_bounds[1]])

    # -------------------------------------- ENSEMBLE I/O ------------------------------------------- #

    def save2pkl(self, path: str) -> int:
        """ Saves ensemble into pickle file at given path.

        Args:
            path: File in which object should be saved.

        Returns:
            0 if saving process was successful, 1 otherwise.
        """
        try:
            attr_dicts = {'hybrid': self.hc.get_attr_dict(),
                          'no_pred': self._no_pred,
                          'clouds': {},
                          'predictions': self._predictions,
                          'verts2node': self.verts2node}
            for key in self._clouds:
                attr_dicts['clouds'][key] = self._clouds[key].get_attr_dict()

            with open(path, 'wb') as f:
                pickle.dump(attr_dicts, f)
            f.close()
        except FileNotFoundError:
            print("Saving was not successful as given path is not valid.")
            return 1
        return 0

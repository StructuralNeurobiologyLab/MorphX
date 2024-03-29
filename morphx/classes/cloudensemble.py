# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

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
        self._pred_num = None

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
    def vertices(self) -> np.ndarray:
        return self.flattened.vertices

    @property
    def labels(self) -> np.ndarray:
        return self.flattened.labels

    @property
    def node_labels(self) -> np.ndarray:
        return self.hc.node_labels

    @property
    def types(self) -> np.ndarray:
        return self.hc.types

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
        """ Transforms the CloudEnsemble to a PointCloud where the object information is saved in the
            obj_bounds method.
        """
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

    @property
    def pred_num(self) -> int:
        if self._pred_num is None:
            self._pred_num = self.get_pred_num()
        return self._pred_num

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

    def remove_nodes(self, labels: List[int]):
        if labels is None or len(labels) == 0:
            return
        _ = self.flattened
        mapping = self._hc.remove_nodes(labels)
        new_verts2node = {}
        for key in self.verts2node:
            if key in mapping:
                new_verts2node[mapping[key]] = self.verts2node[key]
        self._flattened = None
        self._verts2node = new_verts2node
        self.flattened.set_verts2node(new_verts2node)

    def map_labels(self, label_mappings: List):
        self.flattened.map_labels(label_mappings)
        self._hc.map_labels(label_mappings)
        for key in self.clouds:
            cloud = self.get_cloud(key)
            if cloud is None:
                pass
            cloud.map_labels(label_mappings)

    # -------------------------------------- SETTERS ------------------------------------------- #

    def add_cloud(self, cloud: PointCloud, cloud_name: str):
        self._clouds[cloud_name] = cloud
        self.reset_ensemble()

    def remove_cloud(self, cloud_name: str):
        try:
            del self._clouds[cloud_name]
        except KeyError:
            return

    def change_hybrid(self, hybrid: HybridCloud):
        self._hc = hybrid
        self.reset_ensemble()

    def reset_ensemble(self):
        self._flattened = None
        self._verts2node = None

    def add_no_pred(self, obj_names: List[str]):
        for name in obj_names:
            if name not in self._no_pred:
                self._no_pred.append(name)

    def set_predictions(self, predictions: dict):
        self.flattened.set_predictions(predictions)
        self._predictions = self._flattened.predictions

    # -------------------------------------- PREDICTION HANDLING ------------------------------------------- #

    def generate_pred_labels(self, mv: bool = True) -> np.ndarray:
        """ Transfers predictions (gathered for the flattened CloudEnsemble) to the labels of each object in the
            CloudEnsemble.

        Returns:
            Predicted labels of the HybridCloud
        """
        self.flattened.generate_pred_labels(mv)
        if self.flattened.obj_bounds is not None:
            hc_bounds = self.flattened.obj_bounds['hybrid']
            self.hc.set_pred_labels(self.flattened.pred_labels[hc_bounds[0]:hc_bounds[1]])
            for key in self.clouds:
                cloud_bounds = self.flattened.obj_bounds[key]
                self._clouds[key].set_pred_labels(self.flattened.pred_labels[cloud_bounds[0]:cloud_bounds[1]])
        else:
            self.hc.set_pred_labels(self.flattened.pred_labels)
        return self.hc.pred_labels

    def get_pred_num(self) -> int:
        return self.flattened.pred_num

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

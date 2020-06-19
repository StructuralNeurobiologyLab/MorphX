# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import pickle
import numpy as np
from typing import List
from morphx.data import basics
from morphx.processing import objects
from morphx.classes.pointcloud import PointCloud


class PredictionMapper:
    def __init__(self, data_path: str, save_path: str, splitfile: str, datatype: str = 'ce',
                 label_remove: List[int] = None):
        """
        Args:
            data_path: Path to objects saved as pickle files. Existing chunking information would
                be available in the folder 'splitted' at this location.
            save_path: Location where mapped predictions from specific mode should be saved.
            datatype: Type of data encoded in string. 'ce' for CloudEnsembles, 'hc' for HybridClouds.
            splitfile: File with splitting information for the dataset of interest
        """
        self._data_path = os.path.expanduser(data_path)
        if not os.path.exists(self._data_path):
            os.makedirs(self._data_path)
        if save_path is None:
            raise ValueError('There must be a save_path as mapped predictions must be saved.')
        self._save_path = os.path.expanduser(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Load chunks or split dataset into chunks if it was not done already
        if not os.path.exists(splitfile):
            raise ValueError('Previous splitting information must exist in order to map back predictions for chunks.')
        with open(splitfile, 'rb') as f:
            self._splitted_objs = pickle.load(f)
        f.close()
        self._datatype = datatype
        self._curr_obj = None
        self._curr_name = None
        self._label_remove = label_remove

    @property
    def save_path(self):
        return self._save_path

    def map_predictions(self, pred_cloud: PointCloud, mapping_idcs: np.ndarray, obj_name: str, chunk_idx: int,
                        sampling: bool = True):
        """ A processed chunk extracted by a ChunkHandler with the predicted labels can
            then be mapped back to its original chunk and the predictions will be
            saved in :attr:`PointCloud.predictions`.

        Args:
            pred_cloud: Processed cloud with predictions as labels.
            mapping_idcs: The indices of the vertices in the local BFS context from which samples were taken.
            obj_name: The Filename (without .pkl) of the object from which the chunk was extracted.
            chunk_idx: The index of the chunk within the object with name :attr:`obj_name`.
            sampling: Flag whether sampling from the subset was used or not.
        """
        if self._curr_name is None:
            self.load_prediction(obj_name)

        # If requested object differs from object in memory, save current object and try loading new object from save
        # path in case previous predictions were already saved before. If that fails, load new object from data path
        if self._curr_name != obj_name:
            self.save_prediction()
            self.load_prediction(obj_name)
        node_context = self._splitted_objs[obj_name][chunk_idx]
        # Get indices of vertices for requested local BFS
        _, idcs = objects.extract_cloud_subset(self._curr_obj, node_context)
        mapping_idcs = mapping_idcs.astype(int)
        for pred_idx, subset_idx in enumerate(mapping_idcs):
            if sampling:
                # Get indices of vertices in full object (not only in the subset)
                vertex_idx = idcs[subset_idx]
            else:
                # if no sampling was used, the indices can be mapped directly
                vertex_idx = subset_idx
            try:
                self._curr_obj.predictions[vertex_idx].append(int(pred_cloud.labels[pred_idx]))
            except KeyError:
                self._curr_obj.predictions[vertex_idx] = [int(pred_cloud.labels[pred_idx])]

    def load_prediction(self, name: str):
        self._curr_obj = objects.load_obj(self._datatype, f'{self._data_path}{name}.pkl')
        if self._label_remove is not None:
            self._curr_obj.remove_nodes(self._label_remove)
        self._curr_name = name
        if os.path.exists(f'{self._save_path}{name}_preds.pkl'):
            preds = basics.load_pkl(f'{self._save_path}{name}_preds.pkl')
            self._curr_obj.set_predictions(preds)

    def save_prediction(self):
        with open(f'{self._save_path}{self._curr_name}_preds.pkl', 'wb') as f:
            pickle.dump((f'{self._data_path}{self._curr_name}.pkl', self._curr_obj.predictions), f)

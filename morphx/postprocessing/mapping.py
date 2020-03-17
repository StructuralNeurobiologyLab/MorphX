# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import pickle
import numpy as np
from morphx.processing import ensembles, objects
from morphx.classes.pointcloud import PointCloud
from morphx.classes.hybridcloud import HybridCloud


class PredictionMapper:
    def __init__(self, data_path: str, save_path: str, splitfile: str, datatype: str = 'ce'):
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
        if not os.path.exists(f'{save_path}info/'):
            os.makedirs(f'{save_path}info/')

        # Load chunks or split dataset into chunks if it was not done already
        if not os.path.exists(splitfile):
            raise ValueError('Previous splitting information must exist in order to map back predictions for chunks.')
        with open(splitfile, 'rb') as f:
            self._splitted_objs = pickle.load(f)
        f.close()

        self._datatype = datatype
        self._curr_obj = None
        self._curr_name = None

    @property
    def save_path(self):
        return self._save_path

    def map_predictions(self, pred_cloud: PointCloud, mapping_idcs: np.ndarray, obj_name: str, chunk_idx: int):
        """ A processed chunk extracted by a ChunkHandler with the predicted labels can
            then be mapped back to its original chunk and the predictions will be
            saved in :attr:`PointCloud.predictions`.

        Args:
            pred_cloud: Processed cloud with predictions as labels.
            mapping_idcs: The indices of the vertices in the local BFS context from which samples were taken.
            obj_name: The Filename (without .pkl) of the object from which the chunk was extracted.
            chunk_idx: The index of the chunk within the object with name :attr:`obj_name`.
        """
        if self._curr_name is None:
            self.load_prediction(obj_name)

        # If requested object differs from object in memory, save current object and try loading new object from save
        # path in case previous predictions were already saved before. If that fails, load new object from data path
        if self._curr_name != obj_name:
            self.save_prediction(self._curr_name)
            self.load_prediction(obj_name)

        node_context = self._splitted_objs[obj_name][chunk_idx]
        # Get indices of vertices for requested local BFS
        _, idcs = objects.extract_cloud_subset(self._curr_obj, node_context)
        mapping_idcs = mapping_idcs.astype(int)
        for pred_idx, subset_idx in enumerate(mapping_idcs):
            # Get indices of vertices in full object (not only in the subset)
            vertex_idx = idcs[subset_idx]
            try:
                self._curr_obj.predictions[vertex_idx].append(int(pred_cloud.labels[pred_idx]))
            except KeyError:
                self._curr_obj.predictions[vertex_idx] = [int(pred_cloud.labels[pred_idx])]

    def load_prediction(self, name: str):
        try:
            if self._datatype == 'ce':
                self._curr_obj = ensembles.ensemble_from_pkl(f'{self._save_path}{name}.pkl')
            elif self._datatype == 'hc':
                self._curr_obj = HybridCloud()
                self._curr_obj.load_from_pkl(f'{self._save_path}{name}.pkl')
        except FileNotFoundError:
            if self._datatype == 'ce':
                self._curr_obj = ensembles.ensemble_from_pkl(f'{self._data_path}{name}.pkl')
            elif self._datatype == 'hc':
                self._curr_obj = HybridCloud()
                self._curr_obj.load_from_pkl(f'{self._data_path}{name}.pkl')
        self._curr_name = name

    def save_prediction(self, name: str = None, light: bool = False):
        if name is None:
            name = self._curr_name
        self._curr_obj.save2pkl(f'{self._save_path}{name}.pkl')
        if light:
            # Save additional lightweight cloud for fast inspection
            simple_cloud = objects.filter_preds(self._curr_obj)
            simple_cloud.generate_pred_labels()
            simple_cloud.save2pkl(f'{self._save_path}info/{name}_light.pkl')

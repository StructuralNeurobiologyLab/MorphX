# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import pickle
import numpy as np
from scipy.spatial import cKDTree
from morphx.processing import clouds
from morphx.classes.pointcloud import PointCloud


class PredictionMapper:

    def __init__(self,
                 data_path: str,
                 save_path: str,
                 chunk_size: int):
        """
        Args:
            data_path: Path to HybridClouds saved as pickle files. Existing chunking information would
                be available in the folder 'splitted' at this location.
            save_path: Location where mapped predictions from specific mode should be saved.
            chunk_size: Size of the generated chunks. If existing chunking information should be used,
                this must comply with the chunk size used for generating that information.
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
        if not os.path.exists(self._data_path + 'splitted/' + str(chunk_size) + '.pkl'):
            raise ValueError('Previous splitting information must exist in order to map back predictions for chunks.')
        with open(self._data_path + 'splitted/' + str(chunk_size) + '.pkl', 'rb') as f:
            self._splitted_hcs = pickle.load(f)
        f.close()

        self._curr_hc = None
        self._curr_name = None

    @property
    def save_path(self):
        return self._save_path

    def map_predictions(self, pred_cloud: PointCloud, mapping_idcs: np.ndarray, hybrid_name: str, chunk_idx: int):
        """ A processed chunk extracted by a ChunkHandler with the predicted labels can
            then be mapped back to its original chunk and the predictions will be
            saved in :attr:`PointCloud.predictions`.

        Args:
            pred_cloud: Processed cloud with predictions as labels.
            mapping_idcs: The indices of the vertices in the local BFS context from which samples were taken.
            hybrid_name: The Filename (without .pkl) of the HybridCloud from which the chunk was extracted.
            chunk_idx: The index of the chunk within the HybridCloud with name :attr:`hybrid_name`.
        """
        if self._curr_name is None:
            self.load_prediction(hybrid_name)

        # If requested hybrid differs from hybrid in memory, save current hybrid and try loading new hybrid from save
        # path in case previous predictions were already saved before. If that fails, load new hybrid from data path
        if self._curr_name != hybrid_name:
            self.save_prediction(self._curr_name)
            self.load_prediction(hybrid_name)

        local_bfs = self._splitted_hcs[hybrid_name][chunk_idx]

        # Get indices of vertices for requested local BFS
        idcs = []
        for i in local_bfs:
            idcs.extend(self._curr_hc.verts2node[i])

        mapping_idcs = mapping_idcs.astype(int)
        for pred_idx, subset_idx in enumerate(mapping_idcs):
            # Get indices of vertices in full HybridCloud (not only in the subset)
            vertex_idx = idcs[subset_idx]
            self._curr_hc.predictions[vertex_idx].append(int(pred_cloud.labels[pred_idx]))

    def load_prediction(self, name: str):
        try:
            self._curr_hc = clouds.load_cloud(self._save_path + name + '.pkl')
        except FileNotFoundError:
            self._curr_hc = clouds.load_cloud(self._data_path + name + '.pkl')
        self._curr_name = name

    def save_prediction(self, name: str = None):
        if name is None:
            name = self._curr_name
        clouds.save_cloud(self._curr_hc, self._save_path, name=name)

        # Save additional lightweight cloud for fast inspection
        simple_cloud = clouds.filter_preds(self._curr_hc)
        simple_cloud.preds2labels_mv()
        clouds.save_cloud(simple_cloud, self._save_path + 'info/', name=name + '_light')

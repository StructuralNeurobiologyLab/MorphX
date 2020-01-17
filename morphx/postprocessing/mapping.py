# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import pickle
from scipy.spatial import cKDTree
from typing import Callable
from morphx.processing import clouds
from morphx.classes.pointcloud import PointCloud


class PredictionMapper:

    def __init__(self,
                 data_path: str,
                 save_path: str,
                 chunk_size: int,
                 transform: Callable = clouds.Identity()):
        """
        Args:
            data_path: Path to HybridClouds saved as pickle files. Existing chunking information would
                be available in the folder 'splitted' at this location.
            save_path: Location where mapped predictions from specific mode should be saved.
            chunk_size: Size of the generated chunks. If existing chunking information should be used,
                this must comply with the chunk size used for generating that information.
            transform: Transformations which should be applied to the chunks before returning them
                (e.g. see :func:`morphx.processing.clouds.Compose`)
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

        self._transform = transform
        self._curr_hc = None
        self._curr_name = None

    def map_predictions(self, pred_cloud: PointCloud, hybrid_name: str, chunk_idx: int):
        """ A processed chunk extracted by a ChunkHandler with the predicted labels can
            then be mapped back to its original chunk and the predictions will be
            saved in :attr:`PointCloud.predictions`.

        Args:
            pred_cloud: Processed cloud with predictions as labels.
            hybrid_name: The Filename (without .pkl) of the HybridCloud from which the chunk was extracted.
            chunk_idx: The index of the chunk within the HybridCloud with name :attr:`hybrid_name`.
        """
        if self._curr_name is None:
            self._curr_hc = clouds.load_cloud(self._data_path + hybrid_name + '.pkl')
            self._curr_name = hybrid_name

        # If requested hybrid differs from hybrid in memory, save current hybrid and try loading new hybrid from save
        # path in case previous predictions were already saved before. If that fails, load new hybrid from data path
        if self._curr_name != hybrid_name:
            clouds.save_cloud(self._curr_hc, self._save_path, name=self._curr_name)
            try:
                self._curr_hc = clouds.load_cloud(self._save_path + hybrid_name + '.pkl')
            except FileNotFoundError:
                self._curr_hc = clouds.load_cloud(self._data_path + hybrid_name + '.pkl')
            self._curr_name = hybrid_name

        local_bfs = self._splitted_hcs[hybrid_name][chunk_idx]

        # Get indices of vertices for requested local BFS
        idcs = []
        for i in local_bfs:
            idcs.extend(self._curr_hc.verts2node[i])

        # Apply invers transformations to compare the predicted cloud with the original cloud
        if len(pred_cloud.vertices) > 0:
            self._transform(pred_cloud, invers=True)

        # Vertices of predicted cloud can differ from the original ones as sampling is altering the order and may add
        # additional points
        tree = cKDTree(self._curr_hc.vertices[idcs])
        dist, ind = tree.query(pred_cloud.vertices, k=1)
        for pred_idx, vertex_idx in enumerate(ind):
            self._curr_hc.predictions[vertex_idx].append(int(pred_cloud.labels[pred_idx]))

    def save_prediction(self):
        clouds.save_cloud(self._curr_hc, self._save_path, name=self._curr_name)
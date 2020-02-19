# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import torch
import numpy as np
from typing import Union, Tuple
from torch.utils import data
from morphx.processing import clouds
from morphx.data.chunkhandler import ChunkHandler
from morphx.classes.pointcloud import PointCloud


class TorchHandler(data.Dataset):
    """ PyTorch dataset wrapper for underlying chunkhandler dataset. """

    def __init__(self,
                 data_path: str,
                 chunk_size: int,
                 sample_num: int,
                 nclasses: int,
                 transform: clouds.Compose = clouds.Compose([clouds.Identity()]),
                 specific: bool = False,
                 data_type: str = 'ce',
                 obj_feats: dict = None
                 ):
        """ Initializes Dataset. """
        self._ch = ChunkHandler(data_path, chunk_size, sample_num, transform, specific=specific, data_type=data_type,
                                obj_feats=obj_feats)
        self._specific = specific
        self._nclasses = nclasses
        self._sample_num = sample_num

    def __len__(self):
        return len(self._ch)

    def __getitem__(self, item: Union[int, Tuple[str, int]]):
        """ Index gets ignored. """
        # TODO: Improve handling of empty pointclouds
        # Get new sample from base dataloader, skip samples without any points
        ixs = np.empty(0)
        sample = None
        while sample is None:
            if self._specific:
                sample, ixs = self._ch[item]
                # if ixs is None, the requested chunk doesn't exist
                if ixs is None:
                    sample, ixs = PointCloud(vertices=np.zeros((self._sample_num, 3)),
                                             labels=np.zeros(self._sample_num),
                                             features=np.zeros(self._sample_num)), \
                                  np.zeros(self._sample_num)
                    break
            else:
                sample = self._ch[item]

        if sample.labels is not None:
            labels = sample.labels.reshape(len(sample.labels))
        else:
            labels = np.array([])

        # pack all numpy arrays into torch tensors
        pts = torch.from_numpy(sample.vertices).float()
        lbs = torch.from_numpy(labels).long()
        features = torch.from_numpy(sample.features).float()

        if self._specific:
            ixs = torch.from_numpy(ixs)

        no_pred_labels = []
        for name in sample.no_pred:
            no_pred_labels.append(sample.encoding[name])

        # build mask for all indices which should not be used for loss calculation
        idcs = torch.from_numpy(np.isin(sample.labels, no_pred_labels).reshape(-1))
        o_mask = torch.ones(len(sample.vertices), self._nclasses, dtype=torch.bool)
        l_mask = torch.ones(len(sample.vertices), dtype=torch.bool)
        o_mask[idcs] = False
        l_mask[idcs] = False

        if self._specific:
            return {'pts': pts, 'features': features, 'target': lbs, 'o_mask': o_mask, 'l_mask': l_mask, 'map': ixs}
        else:
            return {'pts': pts, 'features': features, 'target': lbs, 'o_mask': o_mask, 'l_mask': l_mask}

    @property
    def obj_names(self):
        return self._ch.obj_names

    def get_hybrid_length(self, name: str):
        return self._ch.get_obj_length(name)

    def get_obj_length(self, name: str):
        return self._ch.get_obj_length(name)

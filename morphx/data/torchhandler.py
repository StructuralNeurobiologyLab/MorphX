# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import torch
import numpy as np
from typing import Callable, Union, Tuple
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
                 transform: clouds.Compose = clouds.Compose([clouds.Identity()]),
                 specific: bool = False):
        """ Initializes Dataset. """
        self.ch = ChunkHandler(data_path, chunk_size, sample_num, transform, specific=specific)
        self.specific = specific

    def __len__(self):
        return len(self.ch)

    def __getitem__(self, item: Union[int, Tuple[str, int]]):
        """ Index gets ignored. """

        # Get new sample from base dataloader, skip samples without any points
        sample = PointCloud(np.array([]))
        while len(sample.vertices) == 0:
            if self.specific:
                sample, centroid = self.ch[item]
            else:
                sample = self.ch[item]

        if sample.labels is not None:
            labels = sample.labels.reshape(len(sample.labels))
        else:
            labels = np.array([])

        # pack all numpy arrays into torch tensors
        pts = torch.from_numpy(sample.vertices).float()
        lbs = torch.from_numpy(labels).long()
        features = torch.ones(len(sample.vertices), 1).float()

        centroid = np.array([0, 0, 0])
        if self.specific:
            centroid = torch.from_numpy(centroid)
            return {'pts': pts, 'features': features, 'target': lbs, 'centroid': centroid}
        else:
            return {'pts': pts, 'features': features, 'target': lbs}

    def hc_names(self):
        return self.ch.hc_names

    def switch_mode(self):
        self.ch.switch_mode()

    def get_hybrid_length(self, name: str):
        return self.ch.get_hybrid_length(name)

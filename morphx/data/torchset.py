# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import torch
from typing import Callable
from torch.utils import data
from morphx.processing import clouds
from morphx.data.cloudset import CloudSet


class TorchSet(data.Dataset):
    """ PyTorch dataset wrapper for underlying pointset dataset. """

    def __init__(self,
                 data_path: str,
                 radius_nm: int,
                 sample_num: int,
                 transform: Callable = clouds.Identity(),
                 iterator_method: str = 'global_bfs',
                 global_source: int = -1,
                 epoch_size: int = 1000,
                 radius_factor: float = 1.5):
        """ Initializes Dataset.

        Args:
            data_path: Absolute path to data.
            radius_nm: The size of the chunks in nanometers.
            sample_num: The number of samples for each chunk.
            transform: Transformations from elektronn3.data.transform.transforms3d which should be applied to incoming
                data.
            iterator_method: The method with which each cell should be iterated.
            global_source: The starting point of the iterator method.
            epoch_size: Size of the data set
            radius_factor: Factor with which radius of global BFS should be calculated. Should be larger than 1, as it
                adjusts the overlap between the cloud chunks
        """

        self.cloudset = CloudSet(data_path, radius_nm, sample_num,
                                 transform=transform,
                                 iterator_method=iterator_method,
                                 global_source=global_source,
                                 epoch_size=epoch_size,
                                 radius_factor=radius_factor)

    def __len__(self):
        return len(self.cloudset)

    def __getitem__(self, index):
        """ Index gets ignored. """

        # get new sample from base dataloader
        sample = self.cloudset[0]
        labels = sample.labels.reshape(sample.labels.shape[0])

        # pack all numpy arrays into torch tensors
        pts = torch.from_numpy(sample.vertices).float()
        lbs = torch.from_numpy(labels).float()
        features = torch.ones(sample.vertices.shape[0], 1).float()

        sample = {
            'pts': pts,
            'feats': features,
            'target': lbs
        }

        return sample

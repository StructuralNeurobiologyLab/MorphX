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
from morphx.classes.hybridcloud import HybridCloud


class TorchSet(data.Dataset):
    """ PyTorch dataset wrapper for underlying pointset dataset. """

    def __init__(self,
                 data_path: str,
                 radius_nm: int,
                 sample_num: int,
                 transform: Callable = clouds.Identity(),
                 iterator_method: str = 'global_bfs',
                 global_source: int = -1,
                 radius_factor: float = 1.5,
                 class_num: int = 2,
                 label_filter: list = None,
                 verbose: bool = False,
                 ensemble: bool = False,
                 size: int = 0):
        """ Initializes Dataset.

        Args:
            data_path: Absolute path to data.
            radius_nm: The size of the chunks in nanometers.
            sample_num: The number of samples for each chunk.
            transform: Transformations from elektronn3.data.transform.transforms3d which should be applied to incoming
                data.
            iterator_method: The method with which each cell should be iterated.
            global_source: The starting point of the iterator method.
            radius_factor: Factor with which radius of global BFS should be calculated. Should be larger than 1, as it
                adjusts the overlap between the cloud chunks
            class_num: Number of classes
                        label_filter: List of labels after which the dataset should be filtered.
            verbose: Enable printing of more detailed messages, __get_item__ will return sample_cloud and local_bfs of
                that current sample.
            ensemble: Enable loading from a pickled CloudEnsemble objects
            size: Leave out analysis step by forwarding the resulting size for the options of this dataset from a
                previous analysis. E.g. if a previous analysis with a radius of 20000 nm gave 2000 pieces, size should
                be 2000.
        """

        self.cloudset = CloudSet(data_path, radius_nm, sample_num,
                                 transform=transform,
                                 iterator_method=iterator_method,
                                 global_source=global_source,
                                 radius_factor=radius_factor,
                                 class_num=class_num,
                                 label_filter=label_filter,
                                 verbose=verbose,
                                 ensemble=ensemble,
                                 size=size)

    def __len__(self):
        return len(self.cloudset)

    def __getitem__(self, index):
        """ Index gets ignored. """

        # get new sample from base dataloader
        sample = self.cloudset[0]

        if sample is None:
            return None

        # TODO: Implement better handling for empty clouds
        # skip samples without any points
        while len(sample.vertices) == 0:
            sample = self.cloudset[0]

        labels = sample.labels.reshape(len(sample.labels))

        # pack all numpy arrays into torch tensors
        pts = torch.from_numpy(sample.vertices).float()
        lbs = torch.from_numpy(labels).long()
        features = torch.ones(len(sample.vertices), 1).float()

        return {'pts': pts, 'features': features, 'target': lbs}

    @property
    def weights(self):
        return self.cloudset.weights

    def activate_single(self, hybrid: HybridCloud):
        self.cloudset.activate_single(hybrid)

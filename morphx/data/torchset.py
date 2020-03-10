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
from morphx.data.merger_cloudset import MergerCloudSet, MergerCloudSetFast
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
                 data_type: str = '',
                 epoch_size = None):
        """ Initializes Dataset.

        Args:
            data_path: Absolute path to data.
            data_type: If it's 'merger' then we use MergerCloudSet, otherwise we use CloudSet
            radius_nm: The size of the chunks in nanometers.
            sample_num: The number of samples for each chunk.
            transform: Transformations from elektronn3.data.transform.transforms3d which should be applied to incoming
                data.
            iterator_method: The method with which each cell should be iterated.
            global_source: The starting point of the iterator method.
            radius_factor: Factor with which radius of global BFS should be calculated. Should be larger than 1, as it
                adjusts the overlap between the cloud chunks
            class_num: Number of classes
        """

        if data_type == 'merger':
            self.cloudset = MergerCloudSetFast(data_path, radius_nm, sample_num,
                                           transform=transform,
                                           iterator_method=iterator_method,
                                           global_source=global_source,
                                           radius_factor=radius_factor,
                                           class_num=class_num,
                                           label_filter=label_filter)
        else:
            self.cloudset = CloudSet(data_path, radius_nm, sample_num,
                                     transform=transform,
                                     iterator_method=iterator_method,
                                     global_source=global_source,
                                     radius_factor=radius_factor,
                                     class_num=class_num,
                                     label_filter=label_filter)

        self.epoch_size = epoch_size

    def __len__(self):
        if self.epoch_size is not None:
            return self.epoch_size
        else:
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


class TorchSetSkeleton(TorchSet):
    def __getitem__(self, index):
        """ Index gets ignored. """

        # get new sample from base dataloader
        sample, chunk_nodes, chunk_node_labels = self.cloudset[0]
        # sample = self.cloudset[0]

        if sample is None:
            return None

        # TODO: Implement better handling for empty clouds
        # skip samples without any points
        while len(sample.vertices) == 0:
            sample = self.cloudset[0]

        # vert_labels = sample.labels.reshape(len(sample.labels))
        node_labels = chunk_node_labels.reshape(len(chunk_node_labels))

        # pack all numpy arrays into torch tensors
        pts = torch.from_numpy(sample.vertices).float()
        nodes = torch.from_numpy(chunk_nodes).float()
        lbs = torch.from_numpy(node_labels).long()
        features = torch.ones(len(sample.vertices), 1).float()
        # vert_labels = torch.from_numpy(vert_labels).long()

        # return {'pts': pts, 'nodes': nodes, 'features': features, 'target': lbs,
        #         'vert_labels': vert_labels}

        return {'pts': pts, 'nodes': nodes, 'features': features, 'target': lbs}
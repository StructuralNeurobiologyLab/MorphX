# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

# Import order for open3d and torch is important
import open3d as o3d
import torch
import time
import numpy as np
from typing import Union, Tuple, List, Optional
from torch.utils import data
from morphx.processing import clouds
from morphx.data.chunkhandler import ChunkHandler
from morphx.classes.pointcloud import PointCloud


class TorchHandler(data.Dataset):
    """ PyTorch dataset wrapper for underlying chunkhandler dataset. """

    def __init__(self,
                 data_path: str,
                 sample_num: int,
                 nclasses: int,
                 feat_dim: int,
                 density_mode: bool = True,
                 bio_density: float = None,
                 tech_density: int = None,
                 chunk_size: int = None,
                 transform: clouds.Compose = clouds.Compose([clouds.Identity()]),
                 specific: bool = False,
                 data_type: str = 'ce',
                 obj_feats: dict = None,
                 label_mappings: List[Tuple[int, int]] = None,
                 hybrid_mode: bool = False,
                 splitting_redundancy: int = 1,
                 label_remove: List[int] = None,
                 sampling: bool = True,
                 force_split: bool = False,
                 padding: int = None,
                 split_on_demand: bool = False,
                 split_jitter: int = 0,
                 epoch_size: int = None,
                 workers: int = 2,
                 voxel_sizes: Optional[dict] = None,
                 ssd_exclude: List[int] = None,
                 ssd_include: List[int] = None,
                 ssd_labels: str = None):
        """ Initializes Dataset. """
        self._ch = ChunkHandler(data_path, sample_num, density_mode=density_mode, bio_density=bio_density,
                                tech_density=tech_density, chunk_size=chunk_size, transform=transform,
                                specific=specific, data_type=data_type, obj_feats=obj_feats,
                                label_mappings=label_mappings, hybrid_mode=hybrid_mode,
                                splitting_redundancy=splitting_redundancy, label_remove=label_remove, sampling=sampling,
                                force_split=force_split, padding=padding, split_on_demand=split_on_demand,
                                split_jitter=split_jitter, epoch_size=epoch_size, workers=workers,
                                voxel_sizes=voxel_sizes, ssd_exclude=ssd_exclude, ssd_include=ssd_include,
                                ssd_labels=ssd_labels)
        self._specific = specific
        self._nclasses = nclasses
        self._sample_num = sample_num
        self._feat_dim = feat_dim
        self._padding = padding

    def __len__(self):
        return len(self._ch)

    def __getitem__(self, item: Union[int, Tuple[str, int]]):
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
                                             features=np.zeros((self._sample_num, self._feat_dim))), \
                                  np.zeros(self._sample_num)
                    break
            else:
                sample = self._ch[item]
                if sample is not None:
                    if len(sample.vertices) == 0:
                        sample = None
                item += 1

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
            if name in sample.encoding.keys():
                no_pred_labels.append(sample.encoding[name])

        # build mask for all indices which should not be used for loss calculation
        idcs = np.isin(sample.labels, no_pred_labels).reshape(-1)
        if self._padding is not None:
            # Mark points which were added as padding for later removal / for ignoring them during loss calculation
            idcs = np.logical_or(idcs, np.all(sample.vertices == self._padding, axis=1))
        idcs = torch.from_numpy(idcs)
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

    @property
    def splitfile(self):
        return self._ch.splitfile

    @property
    def sample_num(self):
        return self._sample_num

    @property
    def num_classes(self):
        return self._nclasses

    def get_hybrid_length(self, name: str):
        return self._ch.get_obj_length(name)

    def get_obj_length(self, name: str):
        return self._ch.get_obj_length(name)

    def get_obj_info(self, name: str, hybrid_only: bool = False):
        return self._ch.get_obj_info(name)

    def get_set_info(self):
        return self._ch.get_set_info()

    def terminate(self):
        self._ch.terminate()

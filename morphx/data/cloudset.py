# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import glob
from typing import Callable
from morphx.processing import graphs, hybrids, clouds
from morphx.classes.hybridcloud import HybridCloud


class CloudSet:
    """ Dataset iterator class that creates point cloud samples from point clouds in pickle files at data_path. """

    def __init__(self,
                 data_path: str,
                 radius_nm: int,
                 sample_num: int,
                 transform: Callable = clouds.Identity(),
                 iterator_method: str = 'global_bfs',
                 global_source: int = -1,
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
            radius_factor: Factor with which radius of global BFS should be calculated. Should be larger than 1, as it
                adjusts the overlap between the cloud chunks
        """

        self.data_path = data_path
        self.radius_nm = radius_nm
        self.sample_num = sample_num
        self.iterator_method = iterator_method
        self.global_source = global_source
        self.transform = transform
        self.radius_factor = radius_factor
        self.files = glob.glob(data_path + '*.pkl')

        # analyse data
        self.size = 0

        # option for single processing
        self.process_single = False

        # options for iterating the dataset
        self.curr_hybrid_idx = 0
        self.curr_node_idx = 0
        self.radius_nm_global = radius_nm*self.radius_factor

        # load first file
        self.curr_hybrid = clouds.load_gt(self.files[self.curr_hybrid_idx])
        self.curr_hybrid.traverser(method=iterator_method, min_dist=self.radius_nm_global, source=self.global_source)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        """ Index gets ignored. """
        # prepare new cell if current one is exhausted
        if self.curr_node_idx >= len(self.curr_hybrid.traverser()):

            # in this case only one hybrid should be processed
            if self.process_single is True:
                # switch back to normal mode
                self.curr_node_idx = 0
                self.process_single = False
                self.curr_hybrid = clouds.load_gt(self.files[self.curr_hybrid_idx])
                self.curr_hybrid.traverser(method=self.iterator_method,
                                           min_dist=self.radius_nm_global,
                                           source=self.global_source)
                return None

            # reset all counters
            self.curr_node_idx = 0
            self.curr_hybrid_idx += 1
            # start over if all cells have been processed
            if self.curr_hybrid_idx >= len(self.files):
                self.curr_hybrid_idx = 0

            # load and prepare new cell
            self.curr_hybrid = clouds.load_gt(self.files[self.curr_hybrid_idx])
            self.curr_hybrid.traverser(method=self.iterator_method, min_dist=self.radius_nm_global,
                                       source=self.global_source)

        # perform local BFS, extract mesh at the respective nodes, sample this set and return it as a point cloud
        spoint = self.curr_hybrid.traverser()[self.curr_node_idx]
        local_bfs = graphs.local_bfs_dist(self.curr_hybrid.graph(), spoint, self.radius_nm)
        subset = hybrids.extract_mesh_subset(self.curr_hybrid, local_bfs)
        sample_cloud = clouds.sample_cloud(subset, self.sample_num)

        # apply transformations from elektronn3.data.transform.transform3d
        aug_cloud = self.transform(sample_cloud)

        # Set pointer to next node of global BFS
        self.curr_node_idx += 1

        return aug_cloud

    def activate_single(self, hybrid: HybridCloud):
        """ Switch cloudset mode to only process the given hybrid """

        self.curr_hybrid = hybrid
        self.curr_hybrid.traverser(method=self.iterator_method,
                                   min_dist=self.radius_nm_global,
                                   source=self.global_source)
        self.process_single = True
        self.curr_node_idx = 0

    def analyse_data(self):
        """ Count number of chunks which can be generated with current settings. """

        print("Analysing data...")
        datasize = 0
        for file in self.files:
            hybrid = clouds.load_gt(file)
            traverser = hybrid.traverser(method=self.iterator_method, min_dist=self.radius_nm_global,
                                         source=self.global_source)
            datasize += len(traverser)
        self.size = datasize

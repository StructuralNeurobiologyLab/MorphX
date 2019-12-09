# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import glob
import numpy as np
from typing import Callable
from morphx.processing import graphs, hybrids, clouds
from morphx.classes.hybridcloud import HybridCloud, PointCloud


class CloudSet:
    """ Dataset iterator class that creates point cloud samples from point clouds in pickle files at data_path. """

    def __init__(self,
                 data_path: str,
                 radius_nm: int,
                 sample_num: int,
                 transform: Callable = clouds.Identity(),
                 iterator_method: str = 'global_bfs',
                 global_source: int = -1,
                 radius_factor: float = 1.5,
                 class_num: int = 2,
                 label_filter: list = None):
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
                adjusts the overlap between the cloud chunks.
            class_num: Number of classes.
            label_filter: List of labels after which the dataset should be filtered.
        """

        self.data_path = data_path
        self.radius_nm = radius_nm
        self.sample_num = sample_num
        self.iterator_method = iterator_method
        self.global_source = global_source
        self.transform = transform
        self.radius_factor = radius_factor
        self.class_num = class_num
        self.label_filter = label_filter

        # find and prepare analysis parameters
        self.files = glob.glob(data_path + '*.pkl')
        self.size = 0
        self._weights = np.ones(class_num)

        # option for single processing
        self.process_single = False

        # options for iterating the dataset
        self.curr_hybrid_idx = 0
        self.curr_node_idx = 0
        self.radius_nm_global = radius_nm*self.radius_factor

        # load first file
        self.curr_hybrid = None
        self.load_new()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        """ Index gets ignored. """
        # prepare new cell if current one is exhausted
        if self.curr_node_idx >= len(self.curr_hybrid.traverser()):

            # process_single finished => switch back to normal when done
            if self.process_single is True:
                self.process_single = False
                self.load_new()
                return None
            else:
                self.load_new()

        # perform local BFS, extract mesh at the respective nodes, sample this set and return it as a point cloud
        spoint = self.curr_hybrid.traverser()[self.curr_node_idx]
        local_bfs = graphs.local_bfs_dist(self.curr_hybrid.graph(), spoint, self.radius_nm)
        subset = hybrids.extract_cloud_subset(self.curr_hybrid, local_bfs)
        sample_cloud = clouds.sample_cloud(subset, self.sample_num)

        # apply transformations
        aug_cloud = self.transform(sample_cloud)

        # Set pointer to next node of global BFS
        self.curr_node_idx += 1

        return aug_cloud

    @property
    def weights(self):
        return self._weights

    def activate_single(self, hybrid: HybridCloud):
        """ Switch cloudset mode to only process the given hybrid

        Args:
            hybrid: The specific hybrid pointcloud which should be processed.
        """

        self.curr_hybrid = hybrid
        self.curr_hybrid.traverser(method=self.iterator_method,
                                   min_dist=self.radius_nm_global,
                                   source=self.global_source)
        self.process_single = True
        self.curr_node_idx = 0

    def load_new(self):
        """ Load next hybrid from dataset and apply possible filters """

        self.curr_hybrid = clouds.load_gt(self.files[self.curr_hybrid_idx])
        if self.label_filter is not None:
            self.curr_hybrid = clouds.filter_labels(self.curr_hybrid, self.label_filter)
        self.curr_hybrid.traverser(method=self.iterator_method,
                                   min_dist=self.radius_nm_global,
                                   source=self.global_source)
        if self.label_filter is not None:
            self.curr_hybrid.filter_traverser()

        self.curr_hybrid_idx += 1
        # start over if all files have been processed
        if self.curr_hybrid_idx >= len(self.files):
            self.curr_hybrid_idx = 0

        self.curr_node_idx = 0

        # load next if current cloud doesn't contain the requested labels
        if len(self.curr_hybrid.traverser()) == 0:
            self.load_new()

    def analyse_data(self):
        """ Count number of chunks which can be generated with current settings and calculate class
            weights based on occurences in dataset. """

        print("Analysing data...")
        # put all clouds together for weight calculation
        total_pc = self.curr_hybrid
        datasize = len(self.curr_hybrid.traverser())

        # iterate remaining files
        for i in range(len(self.files)-1):
            self.load_new()
            total_pc = clouds.merge_clouds(total_pc, self.curr_hybrid)
            datasize += len(self.curr_hybrid.traverser())
        self.size = datasize
        print("Chunking data into {} pieces.".format(datasize))

        self._weights = clouds.calculate_weights_mean(total_pc, self.class_num)

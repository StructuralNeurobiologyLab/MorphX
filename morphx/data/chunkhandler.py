# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import glob
import pickle
import numpy as np
from scipy.spatial import cKDTree
from typing import Callable, Union, Tuple
from morphx.processing import clouds, hybrids
from morphx.preprocessing import splitting
from morphx.classes.pointcloud import PointCloud


class ChunkHandler:
    """ Helper class for loading, sampling and transforming chunks of different HybridClouds. Uses
        :func:`morphx.preprocessing.splitting.split` to generate chunking information (or loads it
        if it is already available for the given chunk_size.

        :attr:`specific` defines two different modes. If the flag is False, the chunks are accessed
        rather randomly, not preserving their original position in the HybridCloud. This mode can be
        used for training purposes. If :attr:`specific` is True, the chunks ca be requested from a
        specific index within a specific HybridCloud. After processing the chunks, the results can
        then be mapped back to that position using :func:`map_predictions`. This mode can be used
        during inference when predictions of chunk are generated and should be inserted into the
        original HybridCloud.
    """

    def __init__(self,
                 data_path: str,
                 chunk_size: int,
                 sample_num: int,
                 transform: Callable = clouds.Identity(),
                 specific: bool = False):
        """
        Args:
            data_path: Path to HybridClouds saved as pickle files. Existing chunking information would
                be available in the folder 'splitted' at this location.
            chunk_size: Size of the generated chunks. If existing chunking information should be used,
                this must comply with the chunk size used for generating that information.
            sample_num: Number of vertices which should be sampled from the surface of each chunk.
            transform: Transformations which should be applied to the chunks before returning them
                (e.g. see :func:`morphx.processing.clouds.Compose`)
            specific: Flag for setting mode of requesting specific or rather randomly drawn chunks.
        """
        self._data_path = os.path.expanduser(data_path)
        if not os.path.exists(self._data_path):
            os.makedirs(self._data_path)

        # Load chunks or split dataset into chunks if it was not done already
        if not os.path.exists(self._data_path + 'splitted/' + str(chunk_size) + '.pkl'):
            splitting.split(data_path, chunk_size)
        with open(self._data_path + 'splitted/' + str(chunk_size) + '.pkl', 'rb') as f:
            self._splitted_hcs = pickle.load(f)
        f.close()

        self._sample_num = sample_num
        self._transform = transform
        self._specific = specific

        # In non-specific mode, the entire dataset gets loaded
        self._hc_names = []
        self._hcs = []
        if not self._specific:
            files = glob.glob(data_path + '*.pkl')
            for file in files:
                slashs = [pos for pos, char in enumerate(file) if char == '/']
                name = file[slashs[-1] + 1:-4]
                self._hc_names.append(name)
                hc = clouds.load_cloud(file)
                self._hcs.append(hc)

        # In specific mode, the files should be loaded sequentially
        self._curr_hc = None
        self._curr_name = None

        # Index of current hc in self.hcs
        self._hc_idx = 0
        # Index of current chunk in current hc
        self._chunk_idx = 0

    def __len__(self):
        """ Depending on the mode either the sum of the number of chunks from each
            HybridClouds gets returned or the number of chunks of the last requested
            HybridCloud in specific mode.
        """
        if self._specific:
            # Size of last requested hybrid
            if self._curr_name is None:
                size = 0
            else:
                size = len(self._splitted_hcs[self._curr_name])
        else:
            # Size of entire dataset
            size = 0
            for name in self._hc_names:
                size += len(self._splitted_hcs[name])
        return size

    def __getitem__(self, item: Union[int, Tuple[str, int]]):
        """ Returns either a chunk from a specific location and HybridCloud or iterates
            the HybridClouds sequentially and returns the chunks in the order of the
            chunking information.

        Args:
            item: With :attr:`specific` as False, this parameter gets ignored. With true
            it must be a tuple of the filename of the requested HybridCloud and the index
            of the chunk within that HybridCloud. E.g. if chunk 5 from HybridCloud in pickle
            file HybridCloud.pkl is requested, this would be ('HybridCloud', 5).
        """
        if self._specific:
            # Get specific item (e.g. chunk 5 of HybridCloud 1)
            if isinstance(item, tuple):
                splitted_hc = self._splitted_hcs[item[0]]

                # In specific mode, the files should be loaded sequentially
                if self._curr_name != item[0]:
                    self._curr_hc = clouds.load_cloud(self._data_path + item[0] + '.pkl')
                    self._curr_name = item[0]

                # Return PointCloud with zeros if requested chunk doesn't exist
                if item[1] >= len(splitted_hc) or abs(item[1]) > len(splitted_hc):
                    return PointCloud(np.zeros((self._sample_num, 3)), np.zeros(self._sample_num))

                # Load local BFS generated by splitter, extract vertices to all nodes of the local BFS (subset) and
                # draw random points of these vertices (sample)
                local_bfs = splitted_hc[item[1]]
                subset = hybrids.extract_cloud_subset(self._curr_hc, local_bfs)
                sample = clouds.sample_cloud(subset, self._sample_num)
            else:
                raise ValueError('In validation mode, items can only be requested with a tuple'
                                 'of HybridCloud name and chunk index within that cloud.')
        else:
            # Get the next item while iterating the entire dataset
            curr_hc_chunks = self._splitted_hcs[self._hc_names[self._hc_idx]]
            if self._chunk_idx >= len(curr_hc_chunks):
                self._hc_idx += 1
                if self._hc_idx >= len(self._hcs):
                    self._hc_idx = 0
                self._chunk_idx = 0

            # Load local BFS generated by splitter, extract vertices to all nodes of the local BFS (subset) and draw
            # random points of these vertices (sample)
            local_bfs = curr_hc_chunks[self._chunk_idx]
            subset = hybrids.extract_cloud_subset(self._hcs[self._hc_idx], local_bfs)
            sample = clouds.sample_cloud(subset, self._sample_num)
            self._chunk_idx += 1

        # Apply transformations (e.g. Composition of Rotation and Normalization)
        if len(sample.vertices) > 0:
            self._transform(sample)

        return sample

    def set_specific_mode(self, specific: bool):
        """ Switch specific mode on and off. """
        self._specific = specific

    def get_hybrid_length(self, name: str):
        """ Returns the number of chunks for a specific HybridCloud.

        Args:
            name: Filename of the requested HybridCloud. If the file is HybridCloud.pkl
            this would be 'HybridCloud'.
        """
        return len(self._splitted_hcs[name])

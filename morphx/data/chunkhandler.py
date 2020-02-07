# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import glob
import pickle
import random
import numpy as np
import ipdb
from typing import Union, Tuple
from morphx.processing import clouds, objects
from morphx.preprocessing import splitting
from morphx.classes.pointcloud import PointCloud


class ChunkHandler:
    """ Helper class for loading, sampling and transforming chunks of different objects. Objects
        must be instances of a MorphX class. Uses :func:`morphx.preprocessing.splitting.split` to
        generate chunking information (or loads it if it is already available for the given chunk_size.

        :attr:`specific` defines two different modes. If the flag is False, the chunks are accessed
        rather randomly, not preserving their original position in the object. This mode can be
        used for training purposes. If :attr:`specific` is True, the chunks ca be requested from a
        specific index within a specific object. After processing the chunks, the results can
        then be mapped back to that position using :func:`map_predictions`. This mode can be used
        during inference when predictions of chunk are generated and should be inserted into the
        original object.
    """

    def __init__(self,
                 data_path: str,
                 chunk_size: int,
                 sample_num: int,
                 transform: clouds.Compose = clouds.Compose([clouds.Identity()]),
                 specific: bool = False):
        """
        Args:
            data_path: Path to objects saved as pickle files. Existing chunking information would
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
            self._splitted_objs = pickle.load(f)
        f.close()

        self._chunk_size = chunk_size
        self._sample_num = sample_num
        self._transform = transform
        self._specific = specific

        # In non-specific mode, the entire dataset gets loaded
        self._obj_names = []
        self._objs = []

        files = glob.glob(data_path + '*.pkl')
        for file in files:
            slashs = [pos for pos, char in enumerate(file) if char == '/']
            name = file[slashs[-1] + 1:-4]
            self._obj_names.append(name)
            if not self._specific:
                obj = objects.load_pkl(file)
                self._objs.append(obj)

        self._chunk_list = []
        if not self._specific:
            for item in self._splitted_objs:
                for idx in range(len(self._splitted_objs[item])):
                    self._chunk_list.append((item, idx))
        random.shuffle(self._chunk_list)

        # In specific mode, the files should be loaded sequentially
        self._curr_obj = None
        self._curr_name = None

        # Index of current chunk
        self._ix = 0

    def __len__(self):
        """ Depending on the mode either the sum of the number of chunks from each
            objects gets returned or the number of chunks of the last requested
            object in specific mode.
        """
        if self._specific:
            # Size of last requested object
            if self._curr_name is None:
                size = 0
            else:
                size = len(self._splitted_objs[self._curr_name])
        else:
            # Size of entire dataset
            size = 0
            for name in self._obj_names:
                size += len(self._splitted_objs[name])
        return size

    def __getitem__(self, item: Union[int, Tuple[str, int]]):
        """ Returns either a chunk from a specific location and object or iterates
            the objects sequentially and returns the chunks in the order of the
            chunking information. If the sampled PointCloud contains no vertices, a
            PointCloud with `self._sample_num` vertices and labels is returned, where
            all numbers are set to 0.

        Args:
            item: With :attr:`specific` as False, this parameter gets ignored. With true
            it must be a tuple of the filename of the requested object and the index
            of the chunk within that object. E.g. if chunk 5 from object in pickle
            file object.pkl is requested, this would be ('object', 5).
        """
        if self._specific:
            # Get specific item (e.g. chunk 5 of object 1)
            if isinstance(item, tuple):
                splitted_obj = self._splitted_objs[item[0]]

                # In specific mode, the files should be loaded sequentially
                if self._curr_name != item[0]:
                    self._curr_obj = objects.load_pkl(self._data_path + item[0] + '.pkl')
                    self._curr_name = item[0]

                # Return PointCloud with zeros if requested chunk doesn't exist
                if item[1] >= len(splitted_obj) or abs(item[1]) > len(splitted_obj):
                    return PointCloud(np.zeros((self._sample_num, 3)), np.zeros(self._sample_num)), \
                           np.zeros(self._sample_num)

                # Load local BFS generated by splitter, extract vertices to all nodes of the local BFS (subset) and
                # draw random points of these vertices (sample)
                local_bfs = splitted_obj[item[1]]
                subset = objects.extract_cloud_subset(self._curr_obj, local_bfs)
                sample, ixs = clouds.sample_objectwise(subset, self._sample_num)
            else:
                raise ValueError('In validation mode, items can only be requested with a tuple of object name and '
                                 'chunk index within that cloud.')
        else:
            # Get the next item while iterating the entire dataset
            if self._ix >= len(self._chunk_list):
                random.shuffle(self._chunk_list)
                self._ix = 0

            next_item = self._chunk_list[self._ix]
            self._ix += 1
            curr_obj_chunks = self._splitted_objs[next_item[0]]
            obj_idx = self._obj_names.index(next_item[0])

            # Load local BFS generated by splitter, extract vertices to all nodes of the local BFS (subset) and draw
            # random points of these vertices (sample)
            local_bfs = curr_obj_chunks[next_item[1]]
            subset = objects.extract_cloud_subset(self._objs[obj_idx], local_bfs)
            sample, ixs = clouds.sample_objectwise(subset, self._sample_num)

        # Apply transformations (e.g. Composition of Rotation and Normalization)
        if len(sample.vertices) > 0:
            self._transform(sample)
        else:
            # Return PointCloud with zeros if sample is empty
            if self._specific:
                return PointCloud(np.zeros((self._sample_num, 3)), np.zeros(self._sample_num)), \
                       np.zeros(self._sample_num)
            else:
                return PointCloud(np.zeros((self._sample_num, 3)), np.zeros(self._sample_num))

        # Return sample and indices from where sample points were taken
        if self._specific:
            return sample, ixs
        else:
            return sample

    @property
    def obj_names(self):
        return self._obj_names

    @property
    def chunk_size(self):
        return self._chunk_size

    @property
    def sample_num(self):
        return self._sample_num

    @property
    def transform(self):
        return self._transform

    def switch_mode(self):
        """ Switch specific mode on and off. """
        self._specific = not self._specific

    def get_obj_length(self, name: str):
        """ Returns the number of chunks for a specific object.

        Args:
            name: Filename of the requested object. If the file is object.pkl this would be 'object'.
        """
        return len(self._splitted_objs[name])

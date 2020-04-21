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
from typing import Union, Tuple, List
from morphx.processing import clouds, objects
from morphx.preprocessing import splitting
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.cloudensemble import CloudEnsemble


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
                 sample_num: int,
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
                 splitting_redundancy: int = 1):
        """
        Args:
            data_path: Path to objects saved as pickle files. Existing chunking information would
                be available in the folder 'splitted' at this location.
            sample_num: Number of vertices which should be sampled from the surface of each chunk.
                Should be equal to the capacity of the given network architecture.
            tech_density: poisson sampling density with which data set was preprocessed in point/um²
            bio_density: chunk sampling density in point/um². This determines the size of the chunks.
                If previous chunking information should be used, this information must be available
                in the splitted/ folder with 'bio_density' as name.
            transform: Transformations which should be applied to the chunks before returning them
                (e.g. see :func:`morphx.processing.clouds.Compose`)
            specific: Flag for setting mode of requesting specific or rather randomly drawn chunks.
            data_type: Type of dataset, 'ce': CloudEnsembles, 'hc': HybridClouds
            obj_feats: Only used when inputs are CloudEnsembles. Dict with feature array (1, n) keyed by
                the name of the corresponding object in the CloudEnsemble. The HybridCloud gets addressed
                with 'hc'.
            label_mappings: list of labels which should get replaced by other labels. E.g. [(1, 2), (3, 2)]
                means that the labels 1 and 3 will get replaced by 3.
            splitting_redundancy: indicates how many times each skeleton node is included in different contexts.
        """
        self._data_path = os.path.expanduser(data_path)
        if not os.path.exists(self._data_path):
            os.makedirs(self._data_path)
        if not os.path.exists(self._data_path + 'splitted/'):
            os.makedirs(self._data_path + 'splitted/')
        self._splitfile = ''
        # Load chunks or split dataset into chunks if it was not done already
        if density_mode:
            if bio_density is None or tech_density is None:
                raise ValueError("Density mode requires bio_density and tech_density")
            self._splitfile = f'{self._data_path}splitted/d{bio_density}_p{sample_num}_r{splitting_redundancy}.pkl'
        else:
            if chunk_size is None:
                raise ValueError("Context mode requires chunk_size.")
            self._splitfile = f'{self._data_path}splitted/s{chunk_size}_r{splitting_redundancy}.pkl'
        self._splitted_objs = None
        if os.path.exists(self._splitfile):
            with open(self._splitfile, 'rb') as f:
                self._splitted_objs = pickle.load(f)
            f.close()
        splitting.split(data_path, self._splitfile, bio_density=bio_density, capacity=sample_num,
                        tech_density=tech_density, density_splitting=density_mode, chunk_size=chunk_size,
                        splitted_hcs=self._splitted_objs, redundancy=splitting_redundancy)
        with open(self._splitfile, 'rb') as f:
            self._splitted_objs = pickle.load(f)
        f.close()

        self._sample_num = sample_num
        self._transform = transform
        self._specific = specific
        self._data_type = data_type
        self._obj_feats = obj_feats
        self._label_mappings = label_mappings
        self._hybrid_mode = hybrid_mode

        # In non-specific mode, the entire dataset gets loaded at once
        self._obj_names = []
        self._objs = []
        files = glob.glob(data_path + '*.pkl')
        for file in files:
            slashs = [pos for pos, char in enumerate(file) if char == '/']
            name = file[slashs[-1] + 1:-4]
            self._obj_names.append(name)
            if not self._specific:
                obj = self._adapt_obj(objects.load_obj(self._data_type, file))
                self._objs.append(obj)
        self._chunk_list = []
        if not self._specific:
            for item in self._splitted_objs:
                if item in self._obj_names:
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
            item: With :attr:`specific` as False, this parameter is a simple integer indexing
                the training examples. With true it must be a tuple of the filename of the
                requested object and the index of the chunk within that object. E.g. if chunk
                5 from object in pickle file object.pkl is requested, this would be ('object', 5).
        """
        if self._specific:
            # Get specific item (e.g. chunk 5 of object 1)
            if isinstance(item, tuple):
                splitted_obj = self._splitted_objs[item[0]]

                # In specific mode, the files should be loaded sequentially
                if self._curr_name != item[0]:
                    self._curr_obj = self._adapt_obj(objects.load_obj(self._data_type,
                                                                      self._data_path + item[0] + '.pkl'))
                    self._curr_name = item[0]

                # Return None if requested chunk doesn't exist
                if item[1] >= len(splitted_obj) or abs(item[1]) > len(splitted_obj):
                    return None, None

                # Load local BFS generated by splitter, extract vertices to all nodes of the local BFS (subset) and
                # draw random points of these vertices (sample)
                local_bfs = splitted_obj[item[1]]
                subset, _ = objects.extract_cloud_subset(self._curr_obj, local_bfs)
                sample, ixs = clouds.sample_cloud(subset, self._sample_num)
            else:
                raise ValueError('In validation mode, items can only be requested with a tuple of object name and '
                                 'chunk index within that cloud.')
        else:
            next_item = self._chunk_list[item]
            curr_obj_chunks = self._splitted_objs[next_item[0]]
            self._curr_obj = self._objs[self._obj_names.index(next_item[0])]

            # Load local BFS generated by splitter, extract vertices to all nodes of the local BFS (subset) and draw
            # random points of these vertices (sample)
            local_bfs = curr_obj_chunks[next_item[1]]
            subset, _ = objects.extract_cloud_subset(self._curr_obj, local_bfs)
            sample, ixs = clouds.sample_cloud(subset, self._sample_num)

        # Apply transformations (e.g. Composition of Rotation and Normalization)
        if len(sample.vertices) > 0:
            self._transform(sample)
        else:
            # Return None if sample is empty, in specific mode return np.empty(0) for idcs to differ from non-existing
            # chunk
            if self._specific:
                return None, np.empty(0)
            else:
                return None

        # Return sample and indices from where sample points were taken
        if self._specific:
            return sample, ixs
        else:
            return sample

    @property
    def obj_names(self):
        return self._obj_names

    @property
    def sample_num(self):
        return self._sample_num

    @property
    def splitfile(self):
        return self._splitfile

    def switch_mode(self):
        """ Switch specific mode on and off. """
        self._specific = not self._specific

    def get_obj_length(self, name: str):
        """ Returns the number of chunks for a specific object.

        Args:
            name: Filename of the requested object. If the file is object.pkl this would be 'object'.
        """
        return len(self._splitted_objs[name])

    def get_obj_info(self, name: str):
        if not self._specific:
            # get objects which are already in cache
            ix = self._obj_names.index(name)
            obj = self._objs[ix]
        else:
            # load objects individually
            obj = self._adapt_obj(objects.load_obj(self._data_type, self._data_path + name + '.pkl'))
        attr_dict = {'vertex_num': len(obj.vertices), 'node_num': len(obj.nodes),
                     'types': list(np.unique(obj.types, return_counts=True)),
                     'labels': list(np.unique(obj.labels, return_counts=True)), 'length': self.get_obj_length(name)}
        return attr_dict

    def get_set_info(self):
        """ Returns a dict with information about the specified dataset. """
        total_attr_dict = {'vertex_num': 0, 'node_num': 0, 'types': [np.array([]), np.array([])],
                           'labels': [np.array([]), np.array([])], 'length': 0}
        for name in self.obj_names:
            attr_dict = self.get_obj_info(name)
            total_attr_dict[name] = attr_dict
            total_attr_dict['vertex_num'] += attr_dict['vertex_num']
            total_attr_dict['node_num'] += attr_dict['node_num']
            total_attr_dict['length'] += attr_dict['length']
            for key in ['labels', 'types']:
                labels = attr_dict[key]
                total_labels = total_attr_dict[key]
                for source_ix, label in enumerate(labels[0]):
                    if label in total_labels[0]:
                        # add label counts of current obj to total
                        target_ix = int(np.argwhere(total_labels[0] == label))
                        total_labels[1][target_ix] += labels[1][source_ix]
                    else:
                        # append label and label counts
                        total_labels[0] = np.append(total_labels[0], int(label))
                        total_labels[1] = np.append(total_labels[1], int(labels[1][source_ix]))
                assert total_attr_dict['labels'][1].sum() == total_attr_dict['vertex_num']
        return total_attr_dict

    def _adapt_obj(self, obj: Union[CloudEnsemble, HybridCloud]) -> Union[CloudEnsemble, HybridCloud]:
        """ Adds given parameters like features or label mappings to the loaded object. """
        # transform to HybridCloud:
        if self._hybrid_mode and isinstance(obj, CloudEnsemble):
            obj = obj.hc
        # change features
        if self._obj_feats is not None:
            if isinstance(obj, CloudEnsemble):
                for name in self._obj_feats:
                    feat_line = self._obj_feats[name]
                    subcloud = obj.get_cloud(name)
                    if subcloud is not None:
                        if isinstance(feat_line, dict):
                            subcloud.types2feat(feat_line)
                        else:
                            feats = np.ones((len(subcloud.vertices), len(feat_line)))
                            feats[:] = feat_line
                            subcloud.set_features(feats)
            elif self._hybrid_mode:
                feats = np.ones(len(obj.vertices)).reshape(-1, 1)
                feats[:] = self._obj_feats['hc']
                obj.set_features(feats)
        # change labels
        if self._label_mappings is not None:
            if isinstance(obj, CloudEnsemble):
                obj.hc.map_labels(self._label_mappings)
                for key in obj.clouds.keys():
                    cloud = obj.get_cloud(key)
                    cloud.map_labels(self._label_mappings)
            else:
                obj.map_labels(self._label_mappings)
        return obj

# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import glob
import pickle
from typing import Callable, Union, Tuple
from morphx.processing import clouds, hybrids
from morphx.preprocessing import splitting


class ChunkLoader:
    def __init__(self,
                 data_path: str,
                 chunk_size: int,
                 sample_num: int,
                 transform: Callable = clouds.Identity(),
                 specific: bool = False):
        self.data_path = os.path.expanduser(data_path)

        # Load chunks or split dataset into chunks if it was not done already
        if not os.path.exists(self.data_path + 'splitted/' + str(chunk_size) + '.pkl'):
            splitting.split(data_path, chunk_size)
        with open(self.data_path + 'splitted/' + str(chunk_size) + '.pkl', 'rb') as f:
            self.splitted_hcs = pickle.load(f)
        f.close()

        self.sample_num = sample_num
        self.transform = transform
        self.specific = specific

        # In training mode, the entire dataset gets loaded
        self.hc_names = []
        self.hcs = []
        if not self.specific:
            files = glob.glob(data_path + '*.pkl')
            for file in files:
                slashs = [pos for pos, char in enumerate(file) if char == '/']
                name = file[slashs[-1] + 1:-4]
                self.hc_names.append(name)
                with open(file, 'rb') as f:
                    hc = clouds.load_cloud(file)
                    self.hcs.append(hc)
                f.close()

        # Index of current hc in self.hcs
        self.hc_idx = 0
        # Index of current chunk in current hc
        self.chunk_idx = 0
        # Size of entire dataset
        self.size = 0
        for name in self.hc_names:
            self.size += len(self.splitted_hcs[name])

    def __len__(self):
        return self.size

    def __getitem__(self, item: Union[int, Tuple[str, int]]):
        # Get specific item (e.g. chunk 5 of HybridCloud 1)
        if self.specific:
            if isinstance(item, tuple):
                if item[0] >= len(self.splitted_hcs) or abs(item[0]) > len(self.splitted_hcs):
                    raise ValueError('This chunk index does not exist in the given HybridCloud.')
                splitted_hc = self.splitted_hcs[item[0]]
                hc_idx = self.hc_names.index(item[0])
                local_bfs = splitted_hc[item[1]]
                subset = hybrids.extract_cloud_subset(self.hcs[hc_idx], local_bfs)
                sample = clouds.sample_cloud(subset, self.sample_num)
            else:
                raise ValueError('In validation mode, items can only be requested with a tuple'
                                 'of HybridCloud name and chunk index within that cloud.')
        # Get the next item while iterating the entire dataset
        else:
            curr_hc_chunks = self.splitted_hcs[self.hc_names[self.hc_idx]]
            if self.chunk_idx >= len(curr_hc_chunks):
                self.hc_idx += 1
                if self.hc_idx >= len(self.hcs):
                    self.hc_idx = 0
                self.chunk_idx = 0

            local_bfs = curr_hc_chunks[self.chunk_idx]
            subset = hybrids.extract_cloud_subset(self.hcs[self.hc_idx], local_bfs)
            sample = clouds.sample_cloud(subset, self.sample_num)
            self.chunk_idx += 1

        if len(sample.vertices) > 0:
            self.transform(sample)

        return sample

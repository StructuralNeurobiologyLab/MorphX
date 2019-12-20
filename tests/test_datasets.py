# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import torch
import pytest
import networkx as nx
from morphx.data.cloudset import CloudSet
from morphx.data.torchset import TorchSet
from morphx.processing import graphs, clouds


@pytest.mark.skip(reason="WIP")
def test_cloudset_sanity():
    wd = os.path.expanduser('~/gt/gt_results/')
    radius_nm = 10000
    sample_num = 100
    data = CloudSet(wd, radius_nm, sample_num, transform=clouds.Center(),
                    label_filter=[1, 3, 4])
    data.analyse_data()
    for i in range(len(data)):
        pc = data[0]


@pytest.mark.skip(reason="WIP")
def test_cloud_traversion():
    wd = os.path.expanduser('~/gt/gt_results/')
    min_dist = 10000
    source = 0
    data = CloudSet(wd, min_dist, 1000, global_source=source)
    data.analyse_data()

    graph = data.curr_hybrid.graph()
    chosen = graphs.global_bfs_dist(graph, min_dist*data.radius_factor, source=source)

    traverser = data.curr_hybrid.traverser()
    print(traverser)

    for item in traverser:
        assert item in chosen

    for item in traverser:
        if item != source:
            path = nx.shortest_path(graph, source=source, target=item, weight='weight')
            dist = 0
            for i in range(len(path) - 1):
                weight = graph[path[i]][path[i + 1]]['weight']
                dist += weight
            assert dist > min_dist * data.radius_factor


@pytest.mark.skip(reason="WIP")
def test_torch_dimensions():
    wd = os.path.expanduser('~/gt/gt_results/')
    min_dist = 10000
    sample_num = 1000
    data = TorchSet(wd, min_dist, sample_num)

    sample = data[0]

    assert sample['pts'].shape == torch.Size([sample_num, 3])
    assert sample['feats'].shape == torch.Size([sample_num, 1])
    assert sample['target'].shape == torch.Size([sample_num])

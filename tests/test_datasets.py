# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import time
import pytest
import networkx as nx
from morphx.data.cloudset import CloudSet
from morphx.data.torchset import TorchSet
from morphx.processing import graphs, clouds


def test_cloudset_sanity():
    wd = os.path.abspath(os.path.dirname(__file__) + '/../example_data/') + '/'
    radius_nm = 10000
    sample_num = 2000
    data = CloudSet(wd, radius_nm, sample_num,
                    transform=clouds.Compose([clouds.Normalization(radius_nm),
                                              clouds.RandomRotate(),
                                              clouds.Center()]))
    for i in range(len(data)):
        pc = data[0]
        assert len(pc.vertices) == sample_num

    radius_nm = 5000
    sample_num = 5000
    data = CloudSet(wd, radius_nm, sample_num,
                    transform=clouds.Compose([clouds.Normalization(radius_nm),
                                              clouds.RandomRotate(),
                                              clouds.Center()]))
    for i in range(len(data)):
        pc = data[0]
        assert len(pc.vertices) == sample_num


def test_torch_sanity():
    wd = os.path.abspath(os.path.dirname(__file__) + '/../example_data/') + '/'
    radius_nm = 20000
    sample_num = 1000
    data = TorchSet(wd, radius_nm, sample_num)

    for i in range(len(data)):
        sample = data[0]
        pts = sample['pts']
        features = sample['features']
        target = sample['target']

        assert len(pts) == sample_num
        assert len(features) == sample_num
        assert len(target) == sample_num


@pytest.mark.skip(reason="WIP")
def test_cloud_traversion():
    wd = os.path.expanduser('~/gt/gt_results/')
    min_dist = 10000
    source = 0
    data = CloudSet(wd, min_dist, 1000, global_source=source)
    data.analyse_data()

    graph = data.curr_hybrid.graph()
    chosen = graphs.global_bfs_dist(graph, min_dist*data.radius_factor, source=source)

    traverser = data.curr_hybrid.base_points()
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


if __name__ == '__main__':
    start = time.time()
    test_cloudset_sanity()
    test_torch_sanity()
    print('Finished after', time.time() - start)

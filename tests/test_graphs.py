# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import time
import pytest
import numpy as np
import networkx as nx

import morphx.processing.objects
from morphx.processing import graphs


# TEST GLOBAL BFS DIST #

def test_global_sanity():
    """
                   6
                   |
    0 -- 1 -- 2 -- 3 -- 4 -- 7
                   |
                   5
                   |
                   8 -- 10 -- 11
                   |
                   9

    min_dist = 2
    """
    g = nx.Graph()
    nodes = np.arange(12)
    g.add_nodes_from(nodes)
    pos = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([2, 0, 0]), np.array([3, 0, 0]),
           np.array([4, 0, 0]), np.array([3, -1, 0]), np.array([3, 1, 0]), np.array([5, 0, 0]),
           np.array([3, -2, 0]), np.array([3, -3, 0]), np.array([4, -2, 0]), np.array([5, -2, 0])]
    for i in range(12):
        g.nodes[i]['position'] = pos[i]
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (3, 6), (4, 7), (5, 8), (8, 9), (8, 10), (10, 11)])
    expected = [0, 2, 4, 8, 11]
    chosen = nodes[graphs.bfs_base_points_euclid(g, 2, source=0)]
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected


def test_radius():
    """
                   6
                   |
    0 -- 1 -- 2 -- 3 -- 4 -- 7
                   |
                   5
                   |
                   8 -- 10 -- 11
                   |
                   9

    min_dist = 3
    """
    g = nx.Graph()
    nodes = np.arange(12)
    g.add_nodes_from(nodes)
    pos = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([2, 0, 0]), np.array([3, 0, 0]),
           np.array([4, 0, 0]), np.array([3, -1, 0]), np.array([3, 1, 0]), np.array([5, 0, 0]),
           np.array([3, -2, 0]), np.array([3, -3, 0]), np.array([4, -2, 0]), np.array([5, -2, 0])]
    for i in range(12):
        g.nodes[i]['position'] = pos[i]
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (3, 6), (4, 7), (5, 8), (8, 9), (8, 10), (10, 11)])
    expected = [0, 3, 9]
    chosen = nodes[graphs.bfs_base_points_euclid(g, 3, source=0)]
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected


# TEST LOCAL BFS DIST #
@pytest.mark.skip(reason="WIP")
def test_local_sanity():
    g = nx.Graph()
    nodes = np.arange(12)
    expected = [5, 4, 0, 6, 10, 7, 9, 8]
    g.add_nodes_from(nodes)
    pos = [np.array([0, 0, 0]), np.array([-0.5, 0, 0]), np.array([-1, 0, 0]), np.array([-0.5, 0, 0]),
           np.array([0.5, 0, 0]), np.array([1, 0, 0]), np.array([1.5, 0, 0]), np.array([1.5, 0, 0]),
           np.array([2, 0, 0]), np.array([2, 0, 0]), np.array([2, 0, 0]), np.array([2.5, 0, 0])]
    for i in range(12):
        g.nodes[i]['position'] = pos[i]
    g.add_edges_from([(0, 1), (1, 2), (0, 3), (0, 4), (4, 5), (5, 6), (5, 7), (7, 8), (7, 9), (6, 10), (10, 11)])
    chosen = nodes[morphx.processing.objects.context_splitting(g, 5, 1)]
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected


# TEST LOCAL BFS NUM #
def test_local_num_sanity():
    g = nx.Graph()
    nodes = np.arange(12)
    expected = [0, 1, 2, 3, 4, 5]
    g.add_nodes_from(nodes)
    g.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (0, 3, 1), (0, 4, 1), (4, 5, 1),
                               (5, 6, 1), (5, 7, 1), (7, 8, 1), (7, 9, 1), (6, 10, 1),
                               (10, 11, 1)])
    chosen = nodes[graphs.bfs_num(g, 0, 5)]
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected
    expected = [0, 1, 3, 4]
    chosen = nodes[graphs.bfs_num(g, 0, 3)]
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected


def test_bfs_iterative():
    g = nx.Graph()
    nodes = np.arange(20)
    g.add_nodes_from(nodes)
    g.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (0, 3, 1), (0, 4, 1), (4, 5, 1), (5, 6, 1), (5, 7, 1), (7, 8, 1),
                               (7, 9, 1), (6, 10, 1), (10, 11, 1), (9, 12, 1), (12, 13, 1), (12, 18, 1), (12, 19, 1),
                               (12, 14, 1), (14, 15, 1), (15, 16, 1), (15, 17, 1)])
    chunks = graphs.bfs_iterative(g, 0, 2)
    expected = [[4, 5], [0, 3], [1, 2], [6, 10], [11], [7, 9], [8], [12, 14], [19], [18], [13], [15, 17], [16]]
    assert len(chunks) == len(expected)
    for chunk in chunks:
        assert chunk in expected

    chunks = graphs.bfs_iterative(g, 9, 4)
    expected = [[12, 14, 19, 18], [13], [9, 7, 5, 8], [4, 0, 1, 3], [2], [6, 10, 11], [15, 17, 16]]
    assert len(chunks) == len(expected)
    for chunk in chunks:
        assert chunk in expected


if __name__ == '__main__':
    start = time.time()
    test_global_sanity()
    test_radius()
    test_local_sanity()
    test_local_num_sanity()
    test_bfs_iterative()
    print('Finished after', time.time() - start)

# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import time
import numpy as np
import networkx as nx
from morphx.processing import graphs


# TEST GLOBAL BFS DIST #
def test_global_sanity():
    g = nx.Graph()
    nodes = np.arange(12)
    expected = [0, 2, 5, 10, 8, 9]

    g.add_nodes_from(nodes)
    g.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (0, 3, 1), (0, 4, 1), (4, 5, 1),
                               (5, 6, 1), (5, 7, 1), (7, 8, 1), (7, 9, 1), (6, 10, 1),
                               (10, 11, 1)])

    chosen = nodes[graphs.global_bfs_dist(g, 2, source=0)]
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected


def test_global_different_weights():
    g = nx.Graph()
    nodes = np.arange(12)
    expected = [0, 1, 2, 4, 6, 7, 9, 11]

    g.add_nodes_from(nodes)
    g.add_weighted_edges_from([(0, 1, 2), (1, 2, 3), (0, 3, 1), (0, 4, 4), (4, 5, 1),
                               (5, 6, 1), (5, 7, 1), (7, 8, 1), (7, 9, 2), (6, 10, 1),
                               (10, 11, 6)])

    chosen = nodes[graphs.global_bfs_dist(g, 2, source=0)]
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected


def test_radius():
    g = nx.Graph()
    nodes = np.arange(12)
    expected1 = [0, 2, 4, 9, 11]
    expected2 = [0, 11]

    g.add_nodes_from(nodes)
    g.add_weighted_edges_from([(0, 1, 2), (1, 2, 3), (0, 3, 1), (0, 4, 4), (4, 5, 1),
                               (5, 6, 1), (5, 7, 1), (7, 8, 1), (7, 9, 2), (6, 10, 1),
                               (10, 11, 6)])

    chosen = nodes[graphs.global_bfs_dist(g, 3.5, source=0)]
    print(chosen)
    assert len(chosen) == len(expected1)
    for item in chosen:
        assert item in expected1

    chosen = nodes[graphs.global_bfs_dist(g, 10.1, source=0)]
    print(chosen)
    assert len(chosen) == len(expected2)
    for item in chosen:
        assert item in expected2


def test_cycles():
    g = nx.Graph()
    nodes = np.arange(9)
    expected = [0, 2, 4, 5]

    g.add_nodes_from(nodes)
    g.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (0, 3, 1), (0, 6, 1), (2, 8, 1),
                               (3, 4, 1), (4, 7, 1), (8, 7, 1), (6, 5, 1)])

    chosen = nodes[graphs.global_bfs_dist(g, 2, source=0)]
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected

    expected = [0]
    g[8][7]['weight'] = 3

    chosen = nodes[graphs.global_bfs_dist(g, 4, source=0)]
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected


# TEST LOCAL BFS DIST #
def test_local_sanity():
    g = nx.Graph()
    nodes = np.arange(12)
    expected = [0, 1, 2, 3, 4, 5]

    g.add_nodes_from(nodes)
    g.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (0, 3, 1), (0, 4, 1), (4, 5, 1),
                               (5, 6, 1), (5, 7, 1), (7, 8, 1), (7, 9, 1), (6, 10, 1),
                               (10, 11, 1)])

    chosen = nodes[graphs.local_bfs_dist(g, 0, 2)]
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected


def test_local_different_weights():
    g = nx.Graph()
    nodes = np.arange(12)
    expected = [0, 1, 3, 4]

    g.add_nodes_from(nodes)
    g.add_weighted_edges_from([(0, 1, 2), (1, 2, 3), (0, 3, 1), (0, 4, 4), (4, 5, 1),
                               (5, 6, 1), (5, 7, 1), (7, 8, 1), (7, 9, 2), (6, 10, 1),
                               (10, 11, 6)])

    chosen = nodes[graphs.local_bfs_dist(g, 0, 4)]
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected


if __name__ == '__main__':
    start = time.time()
    test_global_sanity()
    test_global_different_weights()
    test_radius()
    test_cycles()
    test_local_sanity()
    test_local_different_weights()
    print('Finished after', time.time() - start)
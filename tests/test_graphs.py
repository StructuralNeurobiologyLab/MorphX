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
from morphx.classes.hybridcloud import HybridCloud


# TEST GLOBAL BFS DIST #
def test_global_sanity():
    g = nx.Graph()
    nodes = np.arange(12)
    expected = [0, 2, 5, 10, 8, 9]

    g.add_nodes_from(nodes)

    pos = [np.array([0, 0, 0]), np.array([0.5, 0, 0]), np.array([1, 0, 0]), np.array([0.5, 0, 0]),
           np.array([0.5, 0, 0]), np.array([1, 0, 0]), np.array([1.5, 0, 0]), np.array([1.5, 0, 0]),
           np.array([2, 0, 0]), np.array([2, 0, 0]), np.array([2, 0, 0]), np.array([2.5, 0, 0])]

    for i in range(12):
        g.nodes[i]['position'] = pos[i]

    g.add_edges_from([(0, 1), (1, 2), (0, 3), (0, 4), (4, 5), (5, 6), (5, 7), (7, 8), (7, 9), (6, 10), (10, 11)])

    chosen = nodes[graphs.global_bfs_dist(g, 1, source=0)]
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected


def test_radius():
    """ Same as sanity check, only with 1.5 as min_dist. """
    g = nx.Graph()
    nodes = np.arange(12)
    expected = [0, 6, 7]

    g.add_nodes_from(nodes)

    pos = [np.array([0, 0, 0]), np.array([0.5, 0, 0]), np.array([1, 0, 0]), np.array([0.5, 0, 0]),
           np.array([0.5, 0, 0]), np.array([1, 0, 0]), np.array([1.5, 0, 0]), np.array([1.5, 0, 0]),
           np.array([2, 0, 0]), np.array([2, 0, 0]), np.array([2, 0, 0]), np.array([2.5, 0, 0])]

    for i in range(12):
        g.nodes[i]['position'] = pos[i]

    g.add_edges_from([(0, 1), (1, 2), (0, 3), (0, 4), (4, 5), (5, 6), (5, 7), (7, 8), (7, 9), (6, 10), (10, 11)])

    chosen = nodes[graphs.global_bfs_dist(g, 1.5, source=0)]
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected


# TEST LOCAL BFS DIST #
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

    chosen = nodes[graphs.local_bfs_dist(g, 5, 1)]
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

    chosen = nodes[graphs.local_bfs_num(g, 0, 5)]
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected

    expected = [0, 1, 3, 4]
    chosen = nodes[graphs.local_bfs_num(g, 0, 3)]
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected


# TEST LOCAL BFS NUM #
def test_local_bfs_vertices():
    expected = [0, 1, 4, 5]

    hc = HybridCloud(nodes=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]),
                     edges=np.array([[0, 1], [1, 2], [2, 3], [0, 4], [0, 5]]),
                     vertices=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                                        [2, 2, 2], [3, 3, 3], [3, 3, 3], [3, 3, 3], [4, 4, 4], [4, 4, 4], [4, 4, 4],
                                        [4, 4, 4], [4, 4, 4], [4, 4, 4], [5, 5, 5]]))

    chosen = graphs.local_bfs_vertices(hc, 0, 14)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected

    expected = [0]
    chosen = graphs.local_bfs_vertices(hc, 0, 3)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected

    expected = [1, 2, 3]
    chosen = graphs.local_bfs_vertices(hc, 2, 10)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected


if __name__ == '__main__':
    start = time.time()
    test_global_sanity()
    test_radius()
    test_local_sanity()
    test_local_num_sanity()
    test_local_bfs_vertices()
    print('Finished after', time.time() - start)

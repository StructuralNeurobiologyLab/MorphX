# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch, Philipp Schubert


import os
import pytest
import time
import numpy as np
from morphx.processing import hybrids
from morphx.classes.hybridcloud import HybridCloud
from morphx.data.basics import read_mesh_from_ply, load_skeleton_nx_pkl

test_dir = os.path.dirname(__file__)


def test_hybridcloud_load():
    # TODO: maybe decouple from other tests.
    dir_ex = os.path.abspath(f'{test_dir}/../example_data')
    _, vertices = read_mesh_from_ply(f'{dir_ex}/example_mesh.ply')
    nodes, edges = load_skeleton_nx_pkl(f'{dir_ex}/example_skel.pkl')
    hc = HybridCloud(vertices=vertices, nodes=nodes, edges=edges)
    assert not np.any(np.isnan(hc.vertices))
    assert hc.vertices.ndim == 2, hc.vertices.shape[1] == 3
    return hc


def test_hybridcloud_verts2node():
    # TODO: add value tests
    hc = test_hybridcloud_load()
    _ = hc.verts2node


def test_hybridcloud_pkl_interface():
    hc = test_hybridcloud_load()
    fname = f'{test_dir}/test_hc.pkl'
    try:
        hc.save2pkl(fname)
        hc2 = HybridCloud()
        hc2.load_from_pkl(fname)
        assert hc == hc2
    finally:
        if os.path.isfile(fname):
            os.remove(fname)


def test_node_labels():
    vertices = [[i, i, i] for i in range(10)]
    vertices += vertices
    vertices += vertices

    labels = [i for i in range(10)]
    labels += labels
    labels += labels
    labels = np.array(labels)
    labels[10:20] += 1

    hc = HybridCloud(np.array([[i, i, i] for i in range(10)]), np.array([[i, i+1] for i in range(9)]),
                     vertices=np.array(vertices), labels=labels)

    node_labels = hc.node_labels
    expected = np.array([i for i in range(10)])
    assert np.all(node_labels == expected.reshape((len(expected), 1)))

    node_labels = np.array([1, 2, 1, 1, 2, 2, 2, 1])
    hc = HybridCloud(np.array([[i, i, i] for i in range(8)]), np.array([[i, i+1] for i in range(7)]),
                     vertices=np.array([[1, 1, 1], [2, 2, 2]]), node_labels=node_labels)
    hc.clean_node_labels(2)
    expected = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    assert np.all(hc.node_labels == expected.reshape((len(expected), 1)))


def test_bfs_vertices():
    expected = [0, 1, 4, 5]

    hc = HybridCloud(nodes=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]),
                     edges=np.array([[0, 1], [1, 2], [2, 3], [0, 4], [0, 5]]),
                     vertices=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                                        [2, 2, 2], [3, 3, 3], [3, 3, 3], [3, 3, 3], [4, 4, 4], [4, 4, 4], [4, 4, 4],
                                        [4, 4, 4], [4, 4, 4], [4, 4, 4], [5, 5, 5]]))

    chosen = hybrids.bfs_vertices(hc, 0, 14)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected

    expected = [0]
    chosen = hybrids.bfs_vertices(hc, 0, 3)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected

    expected = [1, 2, 3]
    chosen = hybrids.bfs_vertices(hc, 2, 10)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected


def test_bfs_vertices_diameter():
    hc = HybridCloud(nodes=np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 5, 0], [1, 1, 0], [1, 2, 0], [1, 3, 0],
                                     [1, 5, 0], [-1, 1, 0], [-2, 1, 0], [-3, 1, 0]]),
                     edges=np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10]]),
                     vertices=np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 2, 0], [0, 2, 0], [0, 5, 0],
                                        [0, 5, 0], [1, 1, 0], [1, 1, 0], [1, 2, 0], [1, 3, 0], [1, 5, 0], [-1, 1, 0],
                                        [-2, 1, 0], [-2, 1, 0], [-3, 1, 0]]))

    expected = [0, 1, 2, 4, 8]
    chosen = hybrids.bfs_vertices_diameter(hc, 0, 9, 1)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected

    expected = [0, 1, 2]
    chosen = hybrids.bfs_vertices_diameter(hc, 0, 6, 2)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected

    expected = [1, 2, 5]
    chosen = hybrids.bfs_vertices_diameter(hc, 2, 6, 1)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected


@pytest.mark.skip(reason="WIP")
def test_bfs_vertices_euclid():
    hc = HybridCloud(nodes=np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 5, 0], [1, 1, 0], [1, 2, 0], [1, 3, 0],
                                     [1, 5, 0], [-1, 1, 0], [-2, 1, 0], [-3, 1, 0]]),
                     edges=np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10]]),
                     vertices=np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 2, 0], [0, 2, 0], [0, 5, 0],
                                        [0, 5, 0], [1, 1, 0], [1, 1, 0], [1, 2, 0], [1, 3, 0], [1, 5, 0], [-1, 1, 0],
                                        [-2, 1, 0], [-2, 1, 0], [-3, 1, 0]]))

    expected = [0, 1, 2, 4, 8]
    chosen = hybrids.bfs_vertices_euclid(hc, 0, 9, 3)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected

    expected = [0, 1, 8]
    chosen = hybrids.bfs_vertices_euclid(hc, 0, 4, 3)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected

    expected = [0, 1, 2, 4, 5, 8, 9]
    chosen = hybrids.bfs_vertices_euclid(hc, 0, 20, 3)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected

    expected = [3, 7]
    chosen = hybrids.bfs_vertices_euclid(hc, 3, 5, 2)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected

    expected = [0, 1, 4, 8]
    chosen = hybrids.bfs_vertices_euclid(hc, 0, 8, 20)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected

    expected = [0, 1, 2, 4, 5, 6, 8, 9, 10]
    chosen = hybrids.bfs_vertices_euclid(hc, 0, 20, 4)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected

    hc = HybridCloud(nodes=np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 5, 0], [1, 1, 0], [1, 2, 0], [1, 3, 0],
                                     [1, 5, 0], [-1, 1, 0], [-2, 1, 0], [-3, 1, 0], [2, 1, 0]]),
                     edges=np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
                                     [7, 11]]),
                     vertices=np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 2, 0], [0, 2, 0], [0, 5, 0],
                                        [0, 5, 0], [1, 1, 0], [1, 1, 0], [1, 2, 0], [1, 3, 0], [1, 5, 0], [-1, 1, 0],
                                        [-2, 1, 0], [-2, 1, 0], [-3, 1, 0], [2, 1, 0]]))

    expected = [0, 1, 2, 4, 5, 8, 9]
    chosen = hybrids.bfs_vertices_euclid(hc, 0, 20, 3, cutoff=2)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected

    expected = [0, 1, 2, 8, 9, 10]
    chosen = hybrids.bfs_vertices_euclid(hc, 0, 20, 4, cutoff=1)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected


if __name__ == '__main__':
    start = time.time()
    test_base_points_density()
    print('Finished after', time.time() - start)

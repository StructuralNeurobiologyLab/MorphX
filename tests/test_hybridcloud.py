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
import morphx.processing.objects
from morphx.processing import hybrids
from morphx.classes.hybridcloud import HybridCloud
from morphx.data.basics import read_mesh_from_ply, load_skeleton_nx_pkl

test_dir = os.path.dirname(__file__)


def test_hybridcloud_load():
    # TODO: maybe decouple from other tests.
    dir_ex = os.path.abspath(f'{test_dir}/../data')
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
                     vertices=np.array(vertices), pred_labels=labels)

    pred_node_labels = hc.pred_node_labels
    expected = np.array([i for i in range(10)])
    assert np.all(pred_node_labels == expected.reshape((len(expected), 1)))

    node_labels = np.array([1, 2, 1, 1, 2, 2, 2, 1])
    hc = HybridCloud(np.array([[i, i, i] for i in range(8)]), np.array([[i, i+1] for i in range(7)]),
                     vertices=np.array([[1, 1, 1], [2, 2, 2]]), pred_node_labels=node_labels)
    hc.clean_node_labels(2)
    expected = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    assert np.all(hc.pred_node_labels == expected.reshape((len(expected), 1)))


def test_bfs_vertices():
    expected = [0, 1, 4, 5]

    hc = HybridCloud(nodes=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]),
                     edges=np.array([[0, 1], [1, 2], [2, 3], [0, 4], [0, 5]]),
                     vertices=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                                        [2, 2, 2], [3, 3, 3], [3, 3, 3], [3, 3, 3], [4, 4, 4], [4, 4, 4], [4, 4, 4],
                                        [4, 4, 4], [4, 4, 4], [4, 4, 4], [5, 5, 5]]))

    chosen = morphx.processing.objects.bfs_vertices(hc, 0, 14)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected

    expected = [0]
    chosen = morphx.processing.objects.bfs_vertices(hc, 0, 3)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected

    expected = [1, 2, 3]
    chosen = morphx.processing.objects.bfs_vertices(hc, 2, 10)
    print(chosen)
    assert len(chosen) == len(expected)
    for item in chosen:
        assert item in expected

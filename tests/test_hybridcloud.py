# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch, Philipp Schubert


import os
import numpy as np
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
                     np.array(vertices), labels=labels)

    node_labels = hc.node_labels
    expected = np.array([i for i in range(10)])
    assert np.all(node_labels == expected.reshape((len(expected), 1)))

    node_labels = np.array([1, 2, 1, 1, 2, 2, 2, 1])
    hc = HybridCloud(np.array([[i, i, i] for i in range(8)]), np.array([[i, i+1] for i in range(7)]),
                     np.array([[1, 1, 1], [2, 2, 2]]), node_labels=node_labels)
    hc.clean_node_labels(2)
    expected = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    assert np.all(hc.node_labels == expected.reshape((len(expected), 1)))


# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch, Philipp Schubert


import os
import time
import numpy as np
from morphx.classes.hybridcloud import HybridCloud
from morphx.processing.objects import load_pkl
from morphx.processing.hybrids import extract_mesh_subset


def test_hybridmesh_load():
    p = os.path.abspath(os.path.dirname(__file__) + '/../example_data/example_cell.pkl')
    hm = load_pkl(p)
    return hm


def test_hybridmesh_verts2node():
    hm = test_hybridmesh_load()
    _ = hm.verts2node


def test_hybridmesh_submesh():
    hm = test_hybridmesh_load()
    _ = hm.faces2node
    submesh = extract_mesh_subset(hm, np.arange(len(hm.nodes)))
    assert len(submesh.vertices) == len(hm.vertices)
    assert len(submesh.faces) == len(hm.faces)
    assert np.all(np.sort(submesh.faces.flatten()) == np.sort(hm.faces.flatten()))
    assert np.all(np.sort(submesh.vertices.flatten()) == np.sort(hm.vertices.flatten()))
    _ = extract_mesh_subset(hm, np.arange(min(100, len(hm.nodes))))


def test_hybridmesh_faces2node():
    hm = test_hybridmesh_load()
    _ = hm.faces2node


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


if __name__ == '__main__':
    start = time.time()
    test_hybridmesh_submesh()
    test_node_labels()
    print('Finished after', time.time() - start)

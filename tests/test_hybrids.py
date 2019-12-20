# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch, Philipp Schubert

import numpy as np
import os
from morphx.processing.meshes import load_mesh_gt
from morphx.processing.hybrids import extract_mesh_subset
import time


def test_hybridmesh_load():
    p = os.path.abspath(os.path.dirname(__file__) + '/../example_data/example_cell.pkl')
    hm = load_mesh_gt(p)
    return hm


def test_hybridmesh_vert2skel():
    hm = test_hybridmesh_load()
    _ = hm.vert2skel


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


if __name__ == '__main__':
    start = time.time()
    test_hybridmesh_submesh()
    print('Finished after', time.time() - start)

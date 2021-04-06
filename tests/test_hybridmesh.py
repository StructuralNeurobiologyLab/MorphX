# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch, Philipp Schubert


import os
import numpy as np
from morphx.classes.hybridmesh import HybridMesh
from morphx.processing.hybrids import extract_mesh_subset
from morphx.processing.basics import read_mesh_from_ply, load_skeleton_nx_pkl

test_dir = os.path.dirname(__file__)


def test_sanity():
    hm = HybridMesh()
    assert len(hm.faces) == 0
    assert len(hm.normals) == 0
    assert len(hm.vertices) == 0
    assert len(hm.labels) == 0
    assert len(hm.pred_labels) == 0
    assert len(hm.types) == 0
    assert len(hm.features) == 0
    assert len(hm.nodes) == 0
    assert len(hm.node_labels) == 0
    assert len(hm.pred_node_labels) == 0
    assert len(hm.edges) == 0

    hm = HybridMesh(vertices=np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), labels=np.array([1, 2, 3]),
                    faces=np.array([[0, 1, 2]]))
    assert len(hm.vertices) == 3
    assert len(hm.labels) == 3
    assert len(hm.pred_labels) == len(hm.vertices)
    assert len(hm.types) == 0
    assert len(hm.features) == 0
    assert len(hm.faces) == 1


def test_hybridmesh_load():
    # TODO: maybe decouple from other tests.
    dir_ex = os.path.abspath(f'{test_dir}/../data')
    faces, vertices = read_mesh_from_ply(f'{dir_ex}/example_mesh.ply')
    nodes, edges = load_skeleton_nx_pkl(f'{dir_ex}/example_skel.pkl')
    hm = HybridMesh(nodes=nodes, edges=edges, vertices=vertices,
                    faces=faces)
    assert not np.any(np.isnan(hm.faces))
    assert hm.faces.ndim == 2, hm.faces.shape[1] == 3
    return hm


def test_hybridcloud_pkl_interface():
    hm = test_hybridmesh_load()
    fname = f'{test_dir}/test_hc.pkl'
    try:
        hm.save2pkl(fname)
        hm2 = HybridMesh()
        hm2.load_from_pkl(fname)
        assert hm == hm2
    finally:
        if os.path.isfile(fname):
            os.remove(fname)


def test_hybridmesh_verts2node():
    # TODO: add value tests
    hm = test_hybridmesh_load()
    _ = hm.verts2node


def test_hybridmesh_faces2node():
    # TODO: add value tests
    hm = test_hybridmesh_load()
    _ = hm.faces2node


def test_hybridmesh_submesh():
    # TODO: add value tests
    hm = test_hybridmesh_load()
    _ = hm.faces2node
    submesh = extract_mesh_subset(hm, np.arange(len(hm.nodes)))
    assert len(submesh.vertices) == len(hm.vertices)
    assert len(submesh.faces) == len(hm.faces)
    assert np.all(np.sort(submesh.faces.flatten()) == np.sort(hm.faces.flatten()))
    assert np.all(np.sort(submesh.vertices.flatten()) == np.sort(hm.vertices.flatten()))
    _ = extract_mesh_subset(hm, np.arange(min(100, len(hm.nodes))))


if __name__ == '__main__':
    test_hybridmesh_submesh()

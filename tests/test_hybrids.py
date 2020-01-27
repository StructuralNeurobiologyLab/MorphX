# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch, Philipp Schubert


import os
import time
import numpy as np
from syconn.reps.super_segmentation import *
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.hybridmesh import HybridMesh
from morphx.processing.clouds import load_cloud
from morphx.processing.hybrids import extract_mesh_subset, extract_cloud_subset, test_test_test
from morphx.processing import graphs, hybrids, clouds
from syconn.proc.meshes import mesh2obj_file
from plyfile import PlyData, PlyElement

BASE_DIR = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
global_params.wd = '/wholebrain/songbird/j0126/areaxfs_v6/'


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

import matplotlib.pyplot as pyplot
def write_ply_color(points, labels, filename, num_classes=None, colormap=pyplot.cm.jet):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels) + 1
    else:
        assert (num_classes > np.max(labels))

    vertex = []
    # colors = [pyplot.cm.jet(i/float(num_classes)) for i in range(num_classes)]
    colors = [colormap(i / float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        c = colors[labels[i]]
        c = [int(x * 255) for x in c]
        vertex.append((points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    vertex = np.array(vertex,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)


def test_hybridmesh_load():
    p = os.path.join(ROOT_DIR, 'example_data', 'example_cell.pkl')
    hm = load_cloud(p)
    return hm


def test_hybridmesh_verts2node():
    hm = test_hybridmesh_load()
    _ = hm.verts2node


def test_hybridmesh_submesh():
    import time
    start_time = time.time()
    # hm = test_hybridmesh_load()
    # face2node = hm.faces2node

    ssd = SuperSegmentationDataset()
    cell_obj = ssd.get_super_segmentation_object([1848073])[0]
    mesh = cell_obj.load_mesh('sv')
    cell_obj.load_skeleton()

    ind = mesh[0]
    vert = mesh[1]
    norm = mesh[2]

    labels = np.zeros((vert.reshape(-1, 3).shape[0], 1))

    # initialze HybridMesh
    hm = HybridMesh(nodes=cell_obj.skeleton['nodes'], edges=cell_obj.skeleton['edges'], vertices=vert.reshape(-1, 3),
                    faces=ind.reshape(-1, 3), normals=norm, labels=labels)


    # dest_path = "/wholebrain/scratch/yliu/TMP_hybrids_test/whole_mesh_sso.obj"
    # mesh2obj_file(dest_path, [ind, vert, norm])
    # import pdb
    # pdb.set_trace()

    print("--- loading time for HM: %s seconds ---" % (time.time() - start_time))

    traverser = hm.traverser()
    num_chunks = len(traverser)

    for i in range(num_chunks):
        spoint = int(traverser[i])
        # spoint = 4771 # TEST, this is for example_cell.pkl
        local_bfs = graphs.local_bfs_dist(hm.graph(), spoint, 1.2e2)
        # submesh = hybrids.extract_mesh_subset(hm, local_bfs)
        submesh = hybrids.extract_mesh_subset(hm, local_bfs)

        subset = hybrids.extract_cloud_subset(hm, local_bfs)
        sample_cloud = clouds.sample_cloud(subset, 10000)
        output_path = "/wholebrain/scratch/yliu/TMP_hybrids_test/"

        # ========================
        # output subset points
        # ========================
        write_ply(sample_cloud.vertices, output_path + "test_pts_merger.ply")

        # ========================
        # output submesh
        # ========================
        dest_path = "/wholebrain/scratch/yliu/TMP_hybrids_test/test_merger.obj"
        ind = submesh.faces.flatten().astype(np.int)
        vert = submesh.vertices
        norm = submesh.normals

        import pdb
        pdb.set_trace()

        mesh2obj_file(dest_path, [ind, vert, norm])

    import pdb
    pdb.set_trace()

    submesh = extract_mesh_subset(hm, np.arange(len(hm.nodes)))
    ind = submesh.faces.astype(np.int)
    vert = submesh.vertices
    norm = submesh.normals
    dest_path = "/wholebrain/scratch/yliu/TMP_hybrids_test/test.obj"
    mesh2obj_file(dest_path, [ind, vert, norm])

    import pdb
    pdb.set_trace()

    assert len(submesh.vertices) == len(hm.vertices)
    assert len(submesh.faces) == len(hm.faces)
    assert np.all(np.sort(submesh.faces.flatten()) == np.sort(hm.faces.flatten()))
    assert np.all(np.sort(submesh.vertices.flatten()) == np.sort(hm.vertices.flatten()))

    dest_path = "/wholebrain/scratch/yliu/TMP_hybrids_test/test.obj"
    ind = submesh.faces.astype(np.int)
    vert = submesh.vertices
    norm = submesh.normals

    mesh2obj_file(dest_path, [ind, vert, norm])
    import pdb
    pdb.set_trace()

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
    # test_node_labels()
    print('Finished after', time.time() - start)

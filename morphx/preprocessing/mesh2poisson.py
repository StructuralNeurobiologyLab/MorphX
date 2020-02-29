# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import glob
import trimesh
import numpy as np
from tqdm import tqdm
from morphx.processing import meshes, clouds, graphs, hybrids, ensembles
from morphx.classes.hybridmesh import HybridMesh
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.pointcloud import PointCloud


def process_dataset(input_path: str, output_path: str, tech_density: int):
    """ Converts all objects, saved as pickle files at input_path, into poisson disk sampled HybridClouds and
        saves them at output_path with the same names.

    Args:
        input_path: Path to pickle files with HybridMeshs.
        output_path: Path to folder in which results should be stored.
        tech_density: poisson sampling density in meters unit of the trimesh package
    """
    files = glob.glob(input_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    print("Starting to transform mesh dataset into poisson dataset...")
    for file in tqdm(files):
        process_single_thread([file, output_path, tech_density])


def process_single_thread(args):
    """ Converts single pickle file into poisson disk sampled HybridCloud.

    Args:
        args:
            args[0] - file: The full path to a specific pickle file
            args[1] - output_path: The folder where the result should be stored.
            args[2] - tech_density: poisson sampling density in meters unit of the trimesh package
    """
    file = args[0]
    output_path = args[1]
    tech_density = args[2]
    slashs = [pos for pos, char in enumerate(file) if char == '/']
    name = file[slashs[-1] + 1:-4]
    ce = None
    try:
        hm = HybridMesh()
        hm.load_from_pkl(file)
    except TypeError:
        ce = ensembles.ensemble_from_pkl(file)
        hm = ce.hc
    if not isinstance(hm, HybridMesh) or hm.faces is None:
        raise ValueError("Poisson sampling requires existing faces.")
    mesh = trimesh.Trimesh(vertices=hm.vertices, faces=hm.faces).convert_units('meters', guess=True)
    assert mesh.units == 'meters'
    pc = meshes.sample_mesh_poisson_disk(hm, tech_density*round(mesh.area))
    result = HybridCloud(nodes=hm.nodes, edges=hm.edges, vertices=pc.vertices, labels=pc.labels, encoding=hm.encoding,
                         no_pred=hm.no_pred)
    if ce is None:
        result.save2pkl(output_path + name + '_poisson.pkl')
    else:
        ce.change_hybrid(result)
        for key in ce.clouds:
            cloud = ce.clouds[key]
            mesh = trimesh.Trimesh(vertices=cloud.vertices, faces=cloud.faces).convert_units('meters', guess=True)
            assert mesh.units == 'meters'
            pc = meshes.sample_mesh_poisson_disk(cloud, tech_density*round(mesh.area))
            ce.clouds[key] = PointCloud(vertices=pc.vertices, labels=pc.labels, encoding=pc.encoding,
                                        no_pred=pc.no_pred)
        ce.save2pkl(output_path + name + '_poisson.pkl')


def hybridmesh2poisson(hm: HybridMesh, tech_density: int) -> HybridCloud:
    """ Extracts base points on skeleton of hm with a global BFS, performs local BFS around each base point and samples
        the vertices cloud around the local BFS result with poisson disk sampling. The uniform distances of the samples
        are not guaranteed, as the total sample cloud will always be constructed from two different sample processes
        which have their individual uniform distance.

    Args:
        hm: HybridMesh which should be transformed into a HybridCloud with poisson disk sampled points.
        tech_density: poisson sampling density in meters unit of the trimesh package
    """
    context_size = 50
    offset = 0
    visited = 0
    while visited < len(hm.nodes):
        nodes = hm.nodes[offset:offset+context_size]
        nodes_ix = np.arange(offset, offset+len(nodes))
        offset += context_size
        visited += len(nodes)

    total_pc = None
    intermediate = None
    distance = 1000
    skel2node_mapping = True
    counter = -1
    for base in tqdm(hm.base_points(min_dist=distance)):
        local_bfs = graphs.bfs_euclid_sphere(hm.graph(), source=base, max_dist=distance)
        if skel2node_mapping:
            print("Mapping skeleton to node for further processing. This might take a few minutes...")
            skel2node_mapping = False
        extract = hybrids.extract_mesh_subset(hm, local_bfs)
        if len(hm.faces) == 0:
            continue
        mesh = trimesh.Trimesh(vertices=extract.vertices, faces=extract.faces).convert_units('meters', guess=True)
        assert mesh.units == 'meters'
        pc = meshes.sample_mesh_poisson_disk(extract, tech_density*round(mesh.area))
        if intermediate is None:
            intermediate = pc
        else:
            intermediate = clouds.merge_clouds([intermediate, pc])
        # merging causes processing to slow down => hold speed constant by avoiding merge with large cloud
        counter += 1
        if counter % 50 == 0:
            if total_pc is None:
                total_pc = pc
            else:
                total_pc = clouds.merge_clouds([total_pc, intermediate])
            intermediate = None
    total_pc = clouds.merge_clouds([total_pc, intermediate])
    hc = HybridCloud(hm.nodes, hm.edges, vertices=total_pc.vertices, labels=total_pc.labels, encoding=hm.encoding)
    return hc


if __name__ == '__main__':
    process_single_thread(['/home/john/loc_Bachelorarbeit/gt/gt_meshes/examples/sso_34811392.pkl',
                           '/home/john/loc_Bachelorarbeit/gt/gt_meshes/poisson/', 1000])

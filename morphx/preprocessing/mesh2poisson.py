# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import sys
import glob
import trimesh
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
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
    result = hybridmesh2poisson(hm, tech_density)
    if ce is None:
        result.save2pkl(output_path + name + '_poisson.pkl')
    else:
        ce.change_hybrid(result)
        for key in ce.clouds:
            cloud = ce.clouds[key]
            ce.clouds[key] = hybridmesh2poisson(cloud, tech_density)
        ce.save2pkl(output_path + name + '_poisson.pkl')


def hybridmesh2poisson(hm: HybridMesh, tech_density: int) -> PointCloud:
    """ Extracts base points on skeleton of hm with a global BFS, performs local BFS around each base point and samples
        the vertices cloud around the local BFS result with poisson disk sampling. The uniform distances of the samples
        are not guaranteed, as the total sample cloud will always be constructed from two different sample processes
        which have their individual uniform distance.

    Args:
        hm: HybridMesh which should be transformed into a HybridCloud with poisson disk sampled points.
        tech_density: poisson sampling density in meters unit of the trimesh package
    """

    if hm.nodes is None:
        mesh = trimesh.Trimesh(vertices=hm.vertices, faces=hm.faces).convert_units('meters', guess=True)
        assert mesh.units == 'meters'
        area = round(mesh.area)
        if area == 0:
            return PointCloud()
        else:
            pc = meshes.sample_mesh_poisson_disk(hm, tech_density * area)
        result = PointCloud(vertices=pc.vertices, labels=pc.labels, encoding=hm.encoding, no_pred=hm.no_pred)
    else:
        total = None
        context_size = 10
        skel2node_mapping = True
        chunks = graphs.bfs_iterative(hm.graph(), 0, context_size)
        for chunk in tqdm(chunks):
            chunk = np.array(chunk)
            # At the first iteration the face2node mapping must be done
            if skel2node_mapping:
                print("Mapping faces to node for further processing. This might take a few minutes...")
                skel2node_mapping = False
            extract = hybrids.extract_mesh_subset(hm, chunk)
            if len(hm.faces) == 0:
                continue
            # get the mesh area in trimesh units and use it to determine how many points should be sampled
            with suppress_stdout():
                mesh = trimesh.Trimesh(vertices=extract.vertices, faces=extract.faces)\
                    .convert_units('meters', guess=True)
            assert mesh.units == 'meters'
            area = round(mesh.area)
            if area == 0:
                continue
            else:
                pc = meshes.sample_mesh_poisson_disk(extract, tech_density * area)
            if total is None:
                total = pc
            else:
                total = clouds.merge_clouds([total, pc])
        result = HybridCloud(hm.nodes, hm.edges, vertices=total.vertices, labels=total.labels, encoding=hm.encoding)
    return result


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


if __name__ == '__main__':
    process_single_thread(['/home/john/loc_Bachelorarbeit/gt/gt_meshes/examples/sso_34811392.pkl',
                           '/home/john/loc_Bachelorarbeit/gt/gt_meshes/poisson/', 1000])

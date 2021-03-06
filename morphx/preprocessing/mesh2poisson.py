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
from math import floor
from morphx.processing import meshes, clouds, graphs, hybrids, ensembles
from morphx.classes.hybridmesh import HybridMesh
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.pointcloud import PointCloud


def poissonize_dataset(input_path: str, output_path: str, tech_density: int, obj_factor: float):
    """ Converts all objects, saved as pickle files at input_path, into poisson disk sampled HybridClouds and
        saves them at output_path with the same names.

    Args:
        input_path: Path to pickle files with HybridMeshs.
        output_path: Path to folder in which results should be stored.
        tech_density: poisson sampling density in point/um²
    """
    files = glob.glob(input_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    print("Starting to transform mesh dataset into poisson dataset...")
    for file in tqdm(files):
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1] + 1:-4]
        print(name)
        ce = None
        try:
            hm = HybridMesh()
            hm.load_from_pkl(file)
        except TypeError:
            ce = ensembles.ensemble_from_pkl(file)
            hm = ce.hc
        if not isinstance(hm, HybridMesh) or hm.faces is None:
            raise ValueError("Poisson sampling requires existing faces.")
        result = hybridmesh2poisson(hm, tech_density, obj_factor)
        if ce is None:
            result.save2pkl(output_path + name + '_poisson.pkl')
        else:
            ce.change_hybrid(result)
            ce.save2pkl(output_path + name + '_poisson.pkl')
            for key in ce.clouds:
                cloud = ce.clouds[key]
                print(f"\nProcessing {key}")
                ce.clouds[key] = hybridmesh2poisson(cloud, tech_density, obj_factor)
                ce.save2pkl(output_path + name + '_poisson.pkl')
        ce.reset_ensemble()
        ce.hc.set_verts2node(None)
        ce.save2pkl(output_path + name + '_poisson.pkl')


def hybridmesh2poisson(hm: HybridMesh, tech_density: int, obj_factor: float) -> PointCloud:
    """ If there is a skeleton, it gets split into chunks of approximately equal size. For each chunk
        the corresponding mesh piece gets extracted and gets sampled according to its area. If there
        is no skeleton, the mesh is split into multiple parts, depending on its area. Each part is
        then again sampled based on its area.

    Args:
        hm: HybridMesh which should be transformed into a HybridCloud with poisson disk sampled points.
        tech_density: poisson sampling density in point/um². With tech_density = -1, the number of sampled points
            equals the number of vertices in the given HybridMesh.
    """
    if len(hm.nodes) == 0:
        offset = 0
        mesh = trimesh.Trimesh(vertices=hm.vertices, faces=hm.faces)
        area = mesh.area * 1e-06
        # number of chunks should be relative to area
        chunk_number = round(area / 6)
        if area == 0 or chunk_number == 0:
            return PointCloud()
        total = None
        for i in tqdm(range(int(chunk_number))):
            # process all faces left with last chunk
            if i == chunk_number-1:
                chunk_faces = hm.faces[offset:]
            else:
                chunk_faces = hm.faces[offset:offset + floor(len(hm.faces) // chunk_number)]
                offset += floor(len(hm.faces) // chunk_number)
            chunk_hm = HybridMesh(vertices=hm.vertices, faces=chunk_faces, labels=hm.labels, types=hm.types)
            mesh = trimesh.Trimesh(vertices=chunk_hm.vertices, faces=chunk_hm.faces)
            area = mesh.area * 1e-06
            if tech_density == -1:
                pc = meshes.sample_mesh_poisson_disk(chunk_hm, int(len(chunk_hm.vertices) * obj_factor))
            else:
                pc = meshes.sample_mesh_poisson_disk(chunk_hm, tech_density * area * obj_factor)
            if total is None:
                total = pc
            else:
                total = clouds.merge_clouds([total, pc])
        result = PointCloud(vertices=total.vertices, labels=total.labels, encoding=hm.encoding, no_pred=hm.no_pred,
                            types=total.types)
    else:
        total = None
        intermediate = None
        context_size = 5
        skel2node_mapping = True
        counter = 0
        chunks = graphs.bfs_iterative(hm.graph(), 0, context_size)
        for chunk in tqdm(chunks):
            chunk = np.array(chunk)
            # At the first iteration the face2node mapping must be done
            if skel2node_mapping:
                print("Mapping faces to node for further processing. This might take a while...")
                skel2node_mapping = False
            extract = hybrids.extract_mesh_subset(hm, chunk)
            if len(hm.faces) == 0:
                continue
            # get the mesh area in trimesh units and use it to determine how many points should be sampled
            mesh = trimesh.Trimesh(vertices=extract.vertices, faces=extract.faces)
            area = mesh.area * 1e-06
            if area == 0:
                continue
            else:
                if tech_density == -1:
                    pc = meshes.sample_mesh_poisson_disk(extract, len(extract.vertices))
                else:
                    pc = meshes.sample_mesh_poisson_disk(extract, tech_density * area)
            if intermediate is None:
                intermediate = pc
            else:
                intermediate = clouds.merge_clouds([intermediate, pc])
            # merging slows down process => hold speed constant by reducing merging operations
            counter += 1
            if counter % 50 == 0:
                if total is None:
                    total = intermediate
                else:
                    total = clouds.merge_clouds(([total, intermediate]))
                intermediate = None
        total = clouds.merge_clouds([total, intermediate])
        result = HybridCloud(nodes=hm.nodes, edges=hm.edges, vertices=total.vertices, labels=total.labels,
                             encoding=hm.encoding, no_pred=hm.no_pred, types=total.types)
    return result

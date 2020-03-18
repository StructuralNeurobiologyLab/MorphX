# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import pickle
import numpy as np
from math import ceil
from typing import Optional, Tuple
from morphx.classes.pointcloud import PointCloud
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.classes.hybridmesh import HybridMesh
from morphx.processing import clouds, hybrids


# -------------------------------------- ENSEMBLE SAMPLING -------------------------------------- #


def sample_ensemble(ensemble: CloudEnsemble, vertex_number: int, random_seed: Optional[int] = None) \
        -> Tuple[Optional[PointCloud], np.ndarray]:
    """ Samples ensemble parts with respect to their vertex number. Each cloud in the ensemble gets
        len(cloud.vertices)/len(ensemble.vertices)*vertex_number points (ceiled if possible), where
        len(ensemble.vertices) is just the total number of vertices from all ensemble clouds. The
        samples from the different clouds are merged into one PointCloud, the cloud information
        gets saved in the obj_bounds dict of that PointCloud.

    Args:
        ensemble: The ensemble from whose objects the samples should be drawn.
        vertex_number: The number of requested sample points.
        random_seed: A random seed to make sampling deterministic.

    Returns:
        PointCloud object with ensemble cloud information in obj_bounds. Dict with ensemble object names
        as keys and indices of the samples drawn from this object as np.arrays.
    """
    total = 0
    for key in ensemble.clouds:
        total += len(ensemble.clouds[key].vertices)
    if total == 0:
        return None, np.zeros(0)
    current = 0
    result_ixs = np.zeros((vertex_number, 1))
    samples = []
    names = []
    for key in ensemble.clouds:
        verts = len(ensemble.clouds[key].vertices)/total*vertex_number
        if current + ceil(verts) <= vertex_number:
            verts = ceil(verts)
        sample, ixs = clouds.sample_objectwise(ensemble.clouds[key], verts, random_seed=random_seed)
        result_ixs[current:current+verts] = ixs
        current += verts
        samples.append(sample)
        names.append(key)
    result = clouds.merge_clouds(samples, names)
    return result, result_ixs


# -------------------------------------- ENSEMBLE CONVERSION -------------------------------------- #


def ensemble2pointcloud(ensemble: CloudEnsemble) -> Optional[PointCloud]:
    """ Merges vertices and labels from all clouds in the ensemble into a single PointCloud with the respective
        object boundary information saved in obj_bounds. There can only be one HybridCloud per CloudEnsemble, if
        there is one, the nodes and edges get transferred as well.

    Args:
        ensemble: The CloudEnsemble whose clouds should be merged.
    """
    parts = [ensemble.clouds[key] for key in ensemble.clouds.keys()]
    names = [key for key in ensemble.clouds.keys()]
    merged_clouds = clouds.merge_clouds(parts, names, ignore_hybrids=True)
    merged_clouds.add_no_pred(ensemble.no_pred)
    if ensemble.hc is None:
        return merged_clouds
    else:
        return clouds.merge_clouds([ensemble.hc, merged_clouds], ['hybrid', 'clouds'])


# -------------------------------------- HYBRID EXTRACTION ---------------------------------------- #

def extract_subset(ensemble: CloudEnsemble, nodes: np.ndarray) -> Tuple[PointCloud, np.ndarray]:
    """ Extracts all vertices which are associated with the given nodes by the mapping dict verts2node of
        the ensemble.

    Args:
        ensemble: CloudEnsemble from which the vertices should be extracted.
        nodes: node index array which was generated by a local BFS.

    Returns:
        PointCloud with the respective vertices.
    """
    idcs = []
    for i in nodes:
        idcs.extend(ensemble.verts2node[i])
    obj_bounds = {}
    offset = 0
    idcs = np.array(idcs)
    for key in ensemble.flattened.obj_bounds:
        bounds = ensemble.flattened.obj_bounds[key]
        num = len(idcs[np.logical_and(idcs >= bounds[0], idcs < bounds[1])])
        if num != 0:
            obj_bounds[key] = np.array([offset, offset+num])
            offset += num
    if len(idcs) == 0:
        return PointCloud(vertices=np.array([])), np.array([])
    return PointCloud(vertices=ensemble.flattened.vertices[idcs], labels=ensemble.flattened.labels[idcs],
                      features=ensemble.flattened.features[idcs], obj_bounds=obj_bounds,
                      no_pred=ensemble.flattened.no_pred, encoding=ensemble.flattened.encoding), idcs


# -------------------------------------- ENSEMBLE I/O ------------------------------------------- #


def ensemble_from_pkl(path):
    """ Loads an ensemble from an existing pickle file.

    Args:
        path: File path of pickle file.
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print(f"File with name: {path} was not found at this location.")
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    f.close()

    try:
        h = HybridCloud(**obj['hybrid'])
    except TypeError:
        h = HybridMesh(**obj['hybrid'])

    cloudlist = {}
    for key in obj['clouds']:
        try:
            cloudlist[key] = PointCloud(**obj['clouds'][key])
        except TypeError:
            try:
                cloudlist[key] = HybridCloud(**obj['clouds'][key])
            except TypeError:
                cloudlist[key] = HybridMesh(**obj['clouds'][key])

    # check for empty clouds
    empty = []
    for key in cloudlist:
        if cloudlist[key].vertices is None:
            empty.append(key)
    for key in empty:
        cloudlist.pop(key, None)

    try:
        predictions = obj['predictions']
    except KeyError:
        predictions = None

    try:
        verts2node = obj['verts2node']
    except KeyError:
        verts2node = None

    return CloudEnsemble(cloudlist, h, obj['no_pred'], predictions=predictions, verts2node=verts2node)

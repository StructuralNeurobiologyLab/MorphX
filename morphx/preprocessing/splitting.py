# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import glob
import pickle
import random
import numpy as np
from tqdm import tqdm
from typing import List
import open3d as o3d
from morphx.processing import ensembles, objects
from morphx.classes.hybridcloud import HybridCloud
from morphx.processing.objects import context_splitting_kdt_many


def split_single(hc: HybridCloud, ctx: int, base_node_dst: int, radius: int = None):
    """
    Splits a single HybridCloud into chunks. Selects base nodes by voxelization.

    Args:
        hc: HybridCloud which should get split.
        ctx: context size.
        base_node_dst: distance between base nodes. Corresponds to redundancy or the number of chunks per HybridCloud.
        radius: Extraction radius for splitting. See splitting method for more information.

    Returns:
        The generated chunks.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hc.nodes)
    pcd, idcs = pcd.voxel_down_sample_and_trace(
        base_node_dst, pcd.get_min_bound(), pcd.get_max_bound())
    source_nodes = np.max(idcs, axis=1)
    node_arrs = context_splitting_kdt_many(hc, source_nodes, ctx, radius)
    return node_arrs


def split(data_path: str, filename: str, bio_density: float = None, capacity: int = None, tech_density: int = None,
          density_splitting: bool = True, chunk_size: int = None, splitted_hcs: dict = None, redundancy: int = 1,
          label_remove: List[int] = None, split_jitter: int = 0):
    """
    Splits HybridClouds given as pickle files at data_path into multiple subgraphs and saves that chunking information
    in the new folder 'splitted' as a pickled dict. The dict has filenames of the HybridClouds as keys and lists of
    subgraphs as values. The subgraphs are saved as numpy arrays of the indices of skeleton nodes which belong to the
    respective chunk.

    Splitting is done by drawing a random node as the base node for a subgraph extraction. All nodes included in
    the extracted subgraph get removed from the total nodes. The algorithm continues the splitting by drawing new
    random nodes from the remaining nodes as the new base nodes until all nodes have been included in at least one
    subgraph.

    Args:
        data_path: Path to HybridClouds saved as pickle files.
        filename: the file in which splitting information should get saved.
        tech_density: poisson sampling density with which data set was preprocessed in point/um²
        bio_density: chunk sampling density in point/um²
        capacity: number of points which can get processed with given network architecture
        density_splitting: Flag for switching between density and context modes
        chunk_size: Is only used in context mode. Here, the subgraphs get generated by a certain size.
        splitted_hcs: Existing version of splitting information for updates
        redundancy: Indicates how many iterations of base nodes should get used. 1 means, that base nodes get randomly
            drawn from the remaining nodes until all nodes have been included in at least one subgraph. redundancy = n
            means, that base nodes get randomly drawn until all nodes have been included in subgraphs at least n times.
        label_remove: List of labels indicating which nodes should get removed.
        split_jitter: Adds jitter to the context size of the generated chunks.
    """
    # check validity of method call
    if density_splitting:
        if bio_density is None or tech_density is None or capacity is None:
            raise ValueError("density-based splitting requires bio_density, tech_density and capacity.")
        # calculate number of vertices for extracting max surface area (only used for density-based splitting)
        vert_num = int(capacity * tech_density / bio_density)
    else:
        if chunk_size is None:
            raise ValueError("context-based splitting requires chunk_size.")
    # gather all files at given path
    data_path = os.path.expanduser(data_path)
    files = glob.glob(data_path + '*.pkl')
    if splitted_hcs is None:
        splitted_hcs = {}
    # iterate all files at data_path
    for file in tqdm(files):
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1] + 1:-4]
        # apply splitting algorithm to each file which does not have existing splitting information
        if name in splitted_hcs.keys():
            continue
        print(f"No splitting information found for {name}. Splitting it now...")
        obj = ensembles.ensemble_from_pkl(file)
        # remove labels
        if label_remove is not None:
            obj.remove_nodes(labels=label_remove)
        nodes = np.array(obj.graph().nodes)
        base_points = []
        subgraphs = []
        for i in range(redundancy):
            # prepare mask for filtering subgraph nodes
            mask = np.ones(len(nodes), dtype=bool)
            # existing base nodes should not get chosen as a base node again
            mask[np.isin(nodes, base_points)] = False
            # identify remaining nodes
            remaining_nodes = nodes[mask]
            while len(remaining_nodes) != 0:
                # choose random base node from the remaining nodes
                choice = np.random.choice(remaining_nodes, 1)
                base_points.append(choice[0])
                # extract subgraph around the chosen base node using the specified splitting method
                if density_splitting:
                    subgraph = objects.density_splitting(obj, choice[0], vert_num)
                else:
                    jitter = random.randint(0, split_jitter)
                    subgraph = objects.context_splitting_kdt(obj, choice[0], chunk_size + jitter, radius=1000)
                subgraphs.append(subgraph)
                # remove nodes of the extracted subgraph from the remaining nodes
                mask[np.isin(nodes, subgraph)] = False
                remaining_nodes = nodes[mask]
        # save base points for later viewing in corresponding
        base_points = np.array(base_points)
        slashs = [pos for pos, char in enumerate(filename) if char == '/']
        identifier = filename[slashs[-1] + 1:-4]
        basefile = f'{filename[:slashs[-1]]}/base_points/{identifier}/'
        if not os.path.exists(basefile):
            os.makedirs(basefile)
        with open(f'{basefile}{name}_basepoints.pkl', 'wb') as f:
            pickle.dump(base_points, f)
        # update splitting dict with new subgraphs for current object
        splitted_hcs[name] = subgraphs
        with open(filename, 'wb') as f:
            pickle.dump(splitted_hcs, f)
        f.close()
    return splitted_hcs

# MorphX
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
import os
import pickle
import networkx as nx
from plyfile import PlyData
import numpy as np
from typing import Tuple


def read_mesh_from_ply(fname: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read mesh from .ply file. Assumes triangular polygons -> reshape to (-1, 3).

    Args:
        fname: File name.

    Returns:
        Faces (n, 3), vertices (n, 3).

    """
    assert fname.endswith('.ply')
    with open(fname, 'rb') as f:
        plydata = PlyData.read(f)
        vert = plydata['vertex'].data
        vert = vert.view((np.float32, len(vert.dtype.names))).flatten()
        ind = np.array(plydata['face'].data['vertex_indices'].tolist()).flatten()
    return ind.reshape(-1, 3), vert.reshape(-1, 3)


def load_skeleton_nx_pkl(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expects node coordinates stored as attribute 'position'. Node IDs must be
    contiguous.

    Args:
        path: Path to gpkl file.

    Returns:
        Node coordinates, edges (indices to the node array).
    """
    g = nx.read_gpickle(path)
    try:
        nodes = np.array([g.node[n]['position'] for n in g.nodes()])
    except AttributeError:  # networkx >= 2.4 compatibility
        nodes = np.array([g.nodes[n]['position'] for n in g.nodes()])
    edges = np.array(list(g.edges), dtype=np.int)
    assert np.max(edges) + 1 == len(nodes), "Node IDs are non-contiguous."
    return nodes, edges


def save2pkl(obj: object, path: str, name='object') -> int:
    """ Dumps given object into pickle file at given path.

    Args:
        obj: Object which should be saved.
        path: Folder where the object should be saved to.
        name: Name of file in which the object should be saved.

    Returns:
        0 if saving process was successful, 1 otherwise.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, name + '.pkl')
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        f.close()
    except FileNotFoundError:
        print("Saving was not successful as given path is not valid.")
        return 1
    return 0


def load_pkl(path):
    """ Loads an object from an existing pickle file.

    Args:
        path: File path of pickle file.
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print(f"File with name: {path} was not found at this location.")
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    f.close()
    return obj

# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
import random
import networkx as nx
from collections import deque
from typing import Union, Tuple, Optional, Iterable, List, Dict
from scipy.spatial import cKDTree
from morphx.classes.pointcloud import PointCloud
from morphx.classes.hybridmesh import HybridMesh
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.processing import hybrids, ensembles, clouds, graphs, basics


# -------------------------------------- DISPATCHER METHODS -------------------------------------- #


def extract_cloud_subset(obj: Union[HybridCloud, CloudEnsemble],
                         local_bfs: np.ndarray) -> Tuple[PointCloud, np.ndarray]:
    if isinstance(obj, HybridCloud):
        return hybrids.extract_subset(obj, local_bfs)
    elif isinstance(obj, CloudEnsemble):
        return ensembles.extract_subset(obj, local_bfs)
    else:
        raise ValueError(f'Expected CloudEnsemble or HybridCloud instance but got '
                         f'{type(obj)} instead.')


def filter_preds(obj: Union[HybridCloud, CloudEnsemble]) -> PointCloud:
    if isinstance(obj, CloudEnsemble):
        pc = obj.flattened
        pc.set_predictions(obj.predictions)
        obj = pc
    return clouds.filter_preds(obj)


def load_obj(data_type: str, file: str) -> Union[HybridMesh, HybridCloud, PointCloud, CloudEnsemble]:
    if data_type == 'obj':
        return basics.load_pkl(file)
    if data_type == 'ce':
        return ensembles.ensemble_from_pkl(file)
    if data_type == 'hc':
        hc = HybridCloud()
        return hc.load_from_pkl(file)
    if data_type == 'hm':
        hm = HybridMesh()
        return hm.load_from_pkl(file)
    else:
        pc = PointCloud()
        return pc.load_from_pkl(file)


# -------------------------------------- BFS ALGORITHMS -------------------------------------- #

def density_splitting(obj: Union[HybridCloud, CloudEnsemble], source: int, vertex_max: int, radius: int = 1000) \
        -> np.ndarray:
    """ Traverses the skeleton with a BFS. For each node, all nodes within 'radius' are sorted according to their
        distance. Then these nodes are added to the resulting node array as long as the total number of corresponding
        vertices is still below vertex_max.

    Args:
        obj: MorphX object which contains the skeleton.
        source: The index of the node to start with.
        vertex_max: The vertex threshold after which the BFS is stopped.
        radius: Workaround to be independent from the skeleton microstructure. All nodes within this radius get sorted
            according to their distance and are then added to the result.

    Returns:
        Index array which contains indices of nodes in the hc node array which are part of the bfs result.
    """
    chosen = []
    idx_nodes = np.arange(len(obj.nodes))
    visited = []
    vertex_num = 0
    # traverse all nodes
    de = deque([int(source)])
    while de:
        curr = de.pop()
        # visited is node list for traversing the graph
        if curr not in visited:
            visited.append(curr)
            dia_nodes = idx_nodes[np.linalg.norm(obj.nodes - obj.nodes[curr], axis=1) <= radius]
            tree = cKDTree(obj.nodes[dia_nodes])
            dist, ind = tree.query(obj.nodes[source], k=len(dia_nodes))
            if isinstance(ind, int):
                ind = [ind]
            for ix in ind:
                node = dia_nodes[ix]
                # chosen is node list for gathering all valid nodes
                if node not in chosen:
                    if vertex_num + len(obj.verts2node[node]) <= vertex_max:
                        chosen.append(node)
                        vertex_num += len(obj.verts2node[node])
                    else:
                        return np.array(chosen)
            neighbors = obj.graph().neighbors(curr)
            de.extendleft([i for i in neighbors])
    return np.array(chosen)


def context_splitting(obj: Union[HybridCloud, CloudEnsemble], source: int, max_dist: float,
                      radius: int = 1000) -> np.ndarray:
    """ Traverses the skeleton with a BFS. For each node, all neighboring nodes within 'radius' get added to the
        resulting array if their distance to the 'source' node is below 'max_dist'.

    Args:
        obj: MorphX object which contains the skeleton.
        source: The index of the node to start with.
        max_dist: The maximum (Euclidean) distance (in nm) between the two most distance nodes, which limits the BFS.
        radius: Workaround to be independent from the skeleton microstructure. Radius of sphere around each node
            in which all nodes should get processed.

    Returns:
        np.ndarray with nodes sorted recording to the result of the limited BFS
    """
    idx_nodes = np.arange(len(obj.nodes))
    visited = []
    chosen = []
    de = deque([source])
    while de:
        curr = de.pop()
        if curr not in visited:
            visited.append(curr)
            # get all nodes within radius and check if condition is satisfied
            dia_nodes = idx_nodes[np.linalg.norm(obj.nodes - obj.nodes[curr], axis=1) <= radius]
            tree = cKDTree(obj.nodes[dia_nodes])
            dist, ind = tree.query(obj.nodes[curr], k=len(dia_nodes))
            if isinstance(ind, int):
                ind = [ind]
            for ix in ind:
                node = dia_nodes[ix]
                if node not in chosen:
                    if np.linalg.norm(obj.nodes[source] - obj.nodes[node]) <= max_dist:
                        chosen.append(node)
            neighbors = obj.graph().neighbors(curr)
            de.extendleft([i for i in neighbors if np.linalg.norm(obj.nodes[source] - obj.nodes[i]) <= max_dist])
    return np.array(chosen)


def context_splitting_kdt(obj: Union[HybridCloud, CloudEnsemble], sources: Union[Iterable[int], int], max_dist: float,
                          radius: Optional[int] = None) -> Union[List[np.ndarray], np.ndarray]:
    """ Performs a query ball search (via kd-tree) to find the connected sub-graph within `max_dist` starting at the
    source node in the obj's weighted graph.

    Args:
        obj: The HybridCloud/CloudEnsemble with the graph, nodes and vertices.
        sources: The source nodes.
        max_dist: The maximum distance to the source node (Euclidean distance, i.e. not along the graph!).
        radius: Optionally add edges between nodes within `radius`.

    Returns:
        The nodes within the requested context for every source node - same ordering as `sources`.
    """
    iterable = True
    try:
        iter(sources)
    except TypeError:
        sources = [sources]
        iterable = False
    g = nx.Graph()
    g.add_edges_from(obj.edges)
    ctxs = []
    kdt = cKDTree(obj.nodes)
    # TODO: use query_ball_tree if sources is large
    ixs_nn = kdt.query_ball_point(obj.nodes[sources], max_dist)
    for source, ixs in zip(sources, ixs_nn):
        sg_view = nx.subgraph(g, ixs)
        # create new object for subgraph as nx.subgraph is only a frozen view
        sg = nx.Graph()
        sg.add_edges_from(sg_view.edges)
        # add self add to prevent KeyError if the only node in the graph is the source node.
        if len(ixs) == 1:
            sg.add_edges_from([(ixs[0], ixs[0])])
        if radius is not None:
            kdt = cKDTree(obj.nodes[ixs])
            pairs = kdt.query_pairs(radius)
            # remap to subset of indices
            sg.add_edges_from([(ixs[p[0]], ixs[p[1]]) for p in pairs])
        # remove independent nodes outside max_dist
        ctxs.append(np.array(list(nx.node_connected_component(sg, source))))
    if iterable:
        return ctxs
    else:
        return ctxs[0]


def context_splitting_graph_many(obj: Union[HybridCloud, CloudEnsemble], sources: Iterable[int],
                                 max_dist: float) -> List[list]:
    """ Performs a dijkstra shortest paths on the obj's weighted graph to retrieve the skeleton nodes within `max_dist`
    for every source node ID in `sources`.

    Args:
        obj: The HybridCloud/CloudEnsemble with the graph, nodes and vertices.
        sources: The source nodes.
        max_dist: The maximum distance to the source node (along the graph).

    Returns:
        The nodes within the requested context for every source node - same ordering as `sources`.
    """
    g = obj.graph()
    if isinstance(sources, list) and len(sources) == 1:
        path = nx.single_source_dijkstra_path(g, sources[0], weight='weight', cutoff=max_dist)
        return [list(path.keys())]
    else:
        paths = dict(nx.all_pairs_dijkstra_path(g, weight='weight', cutoff=max_dist))
        return [list(paths[s].keys()) for s in sources]


def bfs_vertices(hc: Union[HybridCloud, CloudEnsemble], source: int, vertex_max: int) -> np.ndarray:
    """ Performs a BFS on a graph until the number of vertices which correspond to
        the currently selected nodes reaches the maximum.

    Args:
        hc: The HybridCloud with the graph, nodes and vertices on which the BFS should be performed
        source: The source node from which the BFS should start.
        vertex_max: The maximum number of vertices after which the BFS should stop.

    Returns:
        np.ndarray with nodes sorted according to the result of the limited BFS
    """
    visited = [source]
    vertex_num = len(hc.verts2node[source])
    neighbors = hc.graph().neighbors(source)
    de = deque([i for i in neighbors])
    while de:
        curr = de.pop()
        if curr not in visited:
            if vertex_num + len(hc.verts2node[curr]) <= vertex_max:
                visited.append(curr)
                vertex_num += len(hc.verts2node[curr])
                neighbors = hc.graph().neighbors(curr)
                de.extendleft([i for i in neighbors if i not in visited])
            else:
                return np.array(visited, dtype=np.int)

    return np.array(visited)


def label_split(obj: Union[HybridCloud, CloudEnsemble], center_l: int) -> List[List]:
    """ Works for graphs which have one central component (labeled with center_l) and multiple subgraphs
        emerging from that component. This method splits such graphs into the central component and subgraphs.
        Each component can thereby contain small label variations, each subgraph is purely labeled by the most
        dominating label in that subgraph.

        Procedure:
        1. Reduce graph to nodes which contain consecutive nodes of only one label.
        2. Find the central component in that reduced graph.
        3. Split the graph into subgraphs by removing the central component and taking the remaining connected
           components.
        4. Label all subgraphs according to the dominating vertex labels within them.

        Example:
        1-1-2-1-2-2-2-3-1-3 with center_l = 2
        1. Reduce to 1-2-1-2-3-1-3
        2. Calculate central component by graphs:
        3. Split into: 1-2-1  2  3-1-3
        4. Relabel: 1-1-1  2  3-3-3  (assuming the 1 and 3 labels have the dominating number of vertices)

    Args:
        obj: The MorphX object which contains the graph and the corresponding vertices
        center_l: The label of the central component

    Returns:
        List of subgraphs as node arrays.
    """
    if isinstance(obj, CloudEnsemble):
        obj = obj.hc
    source = obj.nodes[random.randrange(len(obj.nodes))]
    nodes, edges, labels = reduce2label_components(obj, source)
    red_g = nx.Graph()
    red_g.add_nodes_from([(i, dict(label=labels[i], size=len(nodes[i]))) for i in list(nodes.keys())])
    red_g.add_edges_from(edges)
    subgraphs = graphs.extract_label_subgraphs(red_g, center_l)
    # for each subgraph, get the number of vertices with the specific label and subtract the number of vertices
    # where the label would have to change
    subgraphs_score = []
    for subgraph in subgraphs:
        center_l_verts = 0
        re_verts = 0
        for node in subgraph:
            vert_num = 0
            sub_nodes = nodes[node]
            for sub_node in sub_nodes:
                vert_num += len(obj.verts2node[sub_node])
            if labels[node] == center_l:
                center_l_verts += vert_num
            else:
                re_verts += vert_num
        subgraphs_score.append(center_l_verts-re_verts)
    center_ix = subgraphs_score.index(max(subgraphs_score))
    center = subgraphs[center_ix]
    # remove central component
    for node in center:
        red_g.remove_node(node)
    # get connected components and gather all original nodes (and not the one from the reduced graph)
    ccs = nx.connected_components(red_g)
    result_ccs = []
    for cc in ccs:
        result_cc = []
        for node in cc:
            result_cc.extend(nodes[node])
        result_ccs.append(result_cc)
    return result_ccs


def reduce2label_components(obj: Union[HybridCloud, CloudEnsemble], source: int) -> Tuple[Dict, List, Dict]:
    """ Reduces graph of given object to nodes which contain consecutive nodes of only one label.

    Args:
        obj: MorphX object which contains the skeleton and node_labels
        source: The id of the node where the BFS should start.

    Returns:
        Tuple. First element is a dict of lists of consecutive nodes with only one label, keyed by
        the index of a new, representative node. Second element is a list of edges of the representative
        nodes. Third element is a dict of labels keyed by the representative nodes.
    """
    visited = [source]
    nodes = {0: [source]}
    labels = {0: obj.node_labels[source]}
    edges = []
    neighbors = obj.graph().neighbors(source)
    de = deque([(i, 0, obj.node_labels[source]) for i in neighbors])
    while de:
        curr, r_node, prev = de.pop()
        if curr not in visited:
            visited.append(curr)
            curr_l = obj.node_labels[curr]
            # label transition => create new reduced node
            if curr_l != prev:
                r_node += 1
                while r_node in nodes:
                    r_node += 1
                nodes[r_node] = []
                labels[r_node] = curr_l
                edges.append((r_node-1, r_node))
            nodes[r_node].append(curr)
            neighbors = obj.graph().neighbors(curr)
            de.extendleft([(i, r_node, curr_l) for i in neighbors if i not in visited])
    return nodes, edges, labels

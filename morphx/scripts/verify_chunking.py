import glob
import pickle
import numpy as np
from morphx.classes.hybridcloud import HybridCloud
from morphx.processing import graphs, clouds


def load_hybrids(paths):
    h_set = []
    for path in paths:
        with open(path, "rb") as f:
            info = pickle.load(f)
        hc = HybridCloud(info['skel_nodes'], info['skel_edges'], info['mesh_verts'])
        h_set.append(hc)
    return h_set


if __name__ == '__main__':
    # set paths
    wd = "/path/to/pickle/files"
    dest = "/destination/path"

    # load cloud
    file_paths = glob.glob(wd + '*.pkl', recursive=False)
    hybrids = load_hybrids([file_paths[4]])

    # visualize initial state
    hybrid = hybrids[0]
    # clouds.visualize_clouds([hybrid.vertices], capture=True, path=dest + "initial.png")
    # clouds.visualize_clouds([hybrid.skel_nodes])

    # get information
    graph = hybrid.graph()
    mapping = hybrid.mesh2skel_dict
    nodes = list(graph.nodes)
    vertices = hybrid.vertices

    # radius of local BFS at sampling positions
    radius = 2000

    # perform global bfs
    np.random.shuffle(nodes)
    source = np.random.randint(len(nodes))
    spoints = graphs.global_bfs_dist(graph, source, radius * 2)

    # perform splitting and stack results together
    total = np.array([])
    im_name = 0
    for spoint in spoints:
        local_bfs = graphs.local_bfs_dist(graph, spoint, radius)
        subset = graphs.extract_mesh_subset(local_bfs, vertices, mapping)
        if len(total) == 0:
            total = subset
        else:
            total = np.concatenate([total, subset])
        # im_name += 1
        # clouds.visualize_clouds([total], capture=True, path=dest + "{}_subset.png".format(im_name))

    clouds.visualize_clouds([total])

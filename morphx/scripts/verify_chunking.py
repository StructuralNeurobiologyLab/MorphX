import glob
import pickle
import numpy as np
from morphx.classes.hybridcloud import HybridCloud
from morphx.processing import graphs, clouds, hybrids


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
    wd = "/home/john/wholebrain/wholebrain/u/jklimesch/gt/gt_results/"
    # dest = "/home/john/sampling_results/"

    # load cloud
    file_paths = glob.glob(wd + '*.pkl', recursive=False)
    hybrid_list = load_hybrids([file_paths[4]])

    # visualize initial state
    hybrid = hybrid_list[0]
    clouds.visualize_clouds([hybrid.vertices])
    # clouds.visualize_clouds([hybrid.skel_nodes])

    # radius of local BFS at sampling positions
    radius = 2000

    # get information
    nodes = hybrid.skel_nodes
    vertices = hybrid.vertices
    mapping = hybrid.vert2skel_dict
    graph = hybrid.graph()

    # perform global bfs
    np.random.shuffle(nodes)
    source = np.random.randint(len(nodes))
    spoints = graphs.global_bfs_dist(graph, source, radius * 2)

    print(len(spoints))

    # perform splitting and stack results together
    total = np.array([])
    im_name = 0
    for spoint in spoints:
        local_bfs = graphs.local_bfs_dist(graph, spoint, radius+500)
        subset = hybrids.extract_mesh_subset(vertices, mapping, local_bfs)
        subset = clouds.sample_cloud(subset, 1000)
        if len(total) == 0:
            total = subset
        else:
            total = np.concatenate((total, subset))
        im_name += 1
        # clouds.visualize_clouds([total], capture=True, path=dest + "{}_subset.png".format(im_name))
        print(im_name)

    clouds.visualize_clouds([total])

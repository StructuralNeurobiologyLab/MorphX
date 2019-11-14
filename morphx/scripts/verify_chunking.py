import glob
import pickle
import time
import numpy as np
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.pointcloud import PointCloud
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
    # clouds.visualize_clouds([hybrid.vertices])

    # radius of local BFS at sampling positions
    radius = 2000
    overlap = 500
    sample_num = 1000

    # get information
    nodes = hybrid.skel_nodes
    graph = hybrid.graph()

    # perform global bfs
    np.random.shuffle(nodes)
    source = np.random.randint(len(nodes))
    start = time.time()
    spoints = graphs.global_bfs_dist(graph, source, radius * 2)
    print("Global BFS duration: ", time.time()-start)
    print("Number of sample points: ", len(spoints))

    # perform splitting and stack results together
    total = PointCloud(np.array([]))
    duration = 0
    im_name = 0
    for spoint in spoints:
        start = time.time()
        local_bfs = graphs.local_bfs_dist(graph, spoint, radius+overlap)
        subset = hybrids.extract_mesh_subset(hybrid, local_bfs)
        subset = clouds.sample_cloud(subset, sample_num)
        duration += time.time()-start

        if len(total.vertices) == 0:
            total = subset
        else:
            total = clouds.merge_clouds(total, subset)
        im_name += 1
    print("Mean sample extraction duration: ", duration/im_name)

    clouds.visualize_clouds([total.vertices])

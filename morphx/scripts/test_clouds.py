import pickle
import glob
import morphx.processing.visualize as vis
from morphx.classes.hybridcloud import HybridCloud


def load_hybrids(paths):
    h_set = []
    for path in paths:
        with open(path, "rb") as f:
            info = pickle.load(f)
        hc = HybridCloud(info['skel_nodes'], info['skel_edges'], info['mesh_verts'], labels=info['vert_labels'])
        h_set.append(hc)
    return h_set


if __name__ == '__main__':
    # set paths
    wd = "/home/john/wholebrain/wholebrain/u/jklimesch/gt/gt_results/"

    file_paths = glob.glob(wd + '*.pkl', recursive=False)

    index = 4
    print("Evaluating cell at " + file_paths[index])

    print("Loading...")
    hybrids = load_hybrids([file_paths[index]])
    print("Finished loading.")

    vis.visualize_clouds(hybrids)

import os
import glob
import time
import argparse
from morphx.processing import meshs, graphs, hybrids, visualize


parser = argparse.ArgumentParser()
parser.add_argument('--pa', type=str, required=True, help='Path to mesh pkl files')

args = parser.parse_args()

path = os.path.expanduser(args.pa)

print("Loading...")
files = glob.glob(path + '*.pkl')
mh = meshs.load_mesh_gt(files[0])
print("Local BFS...")
local_bfs = graphs.local_bfs_dist(mh.graph(), 10000, 10000)
print("Sampling...")
start = time.time()
full_pc = hybrids.extract_cloud_subset(mh, local_bfs)
print("Sampling done in: {} seconds".format(time.time()-start))

print("Extract mesh subset...")
start = time.time()
mc, verts, labels = hybrids.extract_mesh_subset(mh, local_bfs)
print("Extraction done in {} seconds".format(time.time()-start))

print("Sample mesh...")
start = time.time()
pc = meshs.sample_mesh_poisson_disk(mc, verts, labels, 5000)
print("Sampling done in {} seconds".format(time.time()-start))

visualize.visualize_parallel([full_pc], [pc])

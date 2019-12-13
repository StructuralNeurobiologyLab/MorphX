import glob
import argparse
from morphx.processing import clouds, visualize

# PARSE PARAMETERS #

parser = argparse.ArgumentParser(description='Validate a network.')

parser.add_argument('--ta', type=str, required=True, help='Path to target files')
parser.add_argument('--pr', type=str, required=True, help='Path to prediction files')

args = parser.parse_args()

target_path = args.ta
pred_path = args.pr

target_files = glob.glob(target_path + '*.pkl')
pred_files = glob.glob(pred_path + '*.pkl')

target_files.sort()
pred_files.sort()

for idx, target_file in enumerate(target_files):
    pred_file = pred_files[idx]
    target = clouds.load_gt(target_file)
    pred = clouds.load_cloud(pred_file)

    visualize.visualize_parallel([target], [pred])

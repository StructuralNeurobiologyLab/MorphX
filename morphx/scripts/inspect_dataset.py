import glob
import os
import random
from morphx.processing import clouds, visualize


if __name__ == '__main__':
    data_path = os.path.expanduser('~/wholebrain/wholebrain/u/jklimesch/gt/gt_results/visualization/')

    files = glob.glob(data_path + '*.pkl')
    random.shuffle(files)

    hc = clouds.load_cloud(files[0])
    visualize.visualize_clouds([hc])

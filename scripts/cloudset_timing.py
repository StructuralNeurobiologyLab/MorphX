import os
import glob
import time
import argparse
from morphx.processing import clouds, visualize
from morphx.data.cloudset import CloudSet

parser = argparse.ArgumentParser()
parser.add_argument('--da', type=str, help='Set data path.')
parser.add_argument('--ra', type=int, default=10000, help='Radius in nanometers.')
parser.add_argument('--sp', type=int, default=1000, help='Number of sample points.')
args = parser.parse_args()

data_path = os.path.expanduser(args.da)
files = glob.glob(data_path + '*.pkl')

radius = args.ra
sample_num = args.sp

data = CloudSet(data_path, radius, sample_num, clouds.Center(), class_num=5, verbose=True)
data.analyse_data()

for i in range(len(data)):
    start = time.time()
    sample = data[0]
    print("Finished in {} seconds.".format(time.time()-start))

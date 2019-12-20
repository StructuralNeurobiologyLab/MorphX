import os
import glob
import pickle
import numpy as np
from getkey import keys
from morphx.processing import clouds, visualize
from morphx.classes.pointcloud import PointCloud


class ViewControl(object):
    """ Viewer class for comparison of ground truth with processed files or for viewing validation and training
        examples. """

    # TODO: Make this class more abstract and more uniform in terms of loading
    def __init__(self, path1: str, save_path: str, path2: str = None, cloudset: bool = False):
        """
        Args:
            path1: path to pickle files (if path2 != None and comparison of ground truth is intended, this must be the
                folder to the ground truth files).
            path2: Set if comparison of ground truth is intended. This should point to the directory of processed files.
        """

        self.cloudset = cloudset
        self.save_path = os.path.expanduser(save_path)

        self.path1 = os.path.expanduser(path1)
        if cloudset:
            self.files1 = glob.glob(path1 + 'h*.pkl')
        else:
            self.files1 = glob.glob(path1 + '*.pkl')
        self.files1.sort()

        if cloudset:
            self.files2 = glob.glob(path1 + 'cloud*.pkl')
        else:
            self.cmp = False
            if path2 is not None:
                self.path2 = os.path.expanduser(path2)
                self.files2 = glob.glob(path2 + '*.pkl')
                self.files2.sort()
                self.cmp = True
            else:
                self.path2 = None

        if cloudset:
            self.load = self.load_cloudset
        else:
            if self.cmp:
                self.load = self.load_cmp
            else:
                self.load = self.load_val

    def start_view(self, idx: int):
        self.load(idx)

    def core_next(self, cloud1: PointCloud, cloud2: PointCloud, save_name):
        """ Visualizes given clouds and waits for user input:
            Right arrow: Next image
            Left arrow: Previous image
            Up arrow: Save images to png
            Down arrow: Change from static to dynamic view (enable interactive view)

        Args:
            cloud1: First cloud to be visualized.
            cloud2: Second cloud to be visualized.
            save_name: Files will be saved to self.save_path + save_name + _1.png / _2.png

        Returns:
            Change for viewing indices.
        """

        key = visualize.visualize_parallel([cloud1], [cloud2], static=True)
        if key == keys.RIGHT:
            return 1
        if key == keys.UP:
            print("Saving to png...")
            path = self.save_path + save_name
            visualize.visualize_clouds([cloud1], capture=True, path=path + '_1.png')
            visualize.visualize_clouds([cloud2], capture=True, path=path + '_2.png')
            return 0
        if key == keys.DOWN:
            print("Displaying interactive view...")
            # display images for interaction
            visualize.visualize_parallel([cloud1], [cloud2])
            return 0
        if key == keys.LEFT:
            return -1
        if key == keys.ENTER:
            print("Aborting inspection...")
            return None

    def load_cloudset(self, idx: int):
        """ Gets executed when cloudset flag is set and visualizes the results from cloudset chunking performed by
            the morphx.data.analyser save_cloudset method.

        Args:
            idx: Index at which file the viewing should start.
        """

        while idx < len(self.files1):
            file = self.files1[idx]
            slashs = [pos for pos, char in enumerate(file) if char == '/']
            filename = file[slashs[-1]:-4]
            print("Viewing: " + filename)

            with open(file, 'rb') as f:
                content = pickle.load(f)

            hybrid_idx = content[0]
            hybrid_file = [file for file in self.files2 if 'cloud_{}'.format(hybrid_idx) in file]
            hybrid = clouds.load_cloud(hybrid_file[0])

            local_bfs = content[1]
            sample = content[2]
            bfs_cloud = visualize.prepare_bfs(hybrid, local_bfs)

            hybrid_bfs = clouds.merge_clouds(hybrid, bfs_cloud)
            res = self.core_next(hybrid_bfs, sample, 'sample_h{}_i{}'.format(hybrid_idx, idx))

            if res is None:
                return
            else:
                idx += res

    def load_val(self, idx: int):
        """ Method for viewing validation or training examples. These examples must be saved as a list in the pickle
            file and should alternate between target and prediction. E.g. [targetcloud1, predictedcloud1, targetcloud2
            predictedcloud2, ...]
        """

        reverse = False

        # TODO: Make this pythonic and less ugly
        while idx < len(self.files1):
            file = self.files1[idx]
            slashs = [pos for pos, char in enumerate(file) if char == '/']
            filename = file[slashs[-1]:-4]
            print("Viewing: " + filename)

            with open(file, 'rb') as f:
                results = pickle.load(f)
            if reverse:
                i = int(len(results)) - 2
            else:
                i = 0
            while i < int(len(results)):
                orig = results[i]
                pred = results[i+1]
                pred = PointCloud(pred.vertices, np.argmax(pred.vertices, axis=1), encoding=pred.encoding)
                res = self.core_next(orig, pred, filename + '_i{}'.format(i))
                if res is None:
                    return
                i += 2*res
                if res < 0:
                    reverse = True
                    if i < 0:
                        break
                else:
                    reverse = False
            if reverse:
                idx -= 1
            else:
                idx += 1

    def load_cmp(self, idx: int):
        """ Method for comparing ground truth with processed data. Ground truth must be given in first file list,
            processed files in the second file list.
        """

        while idx < len(self.files1):
            gt_file = self.files1[idx]
            pred_file = self.files2[idx]

            slashs = [pos for pos, char in enumerate(gt_file) if char == '/']
            filename = gt_file[slashs[-1]+1:-4]

            gt = clouds.load_gt(gt_file)
            pred = clouds.load_cloud(pred_file)

            res = self.core_next(gt, pred, filename)
            if res is None:
                return
            else:
                idx += res

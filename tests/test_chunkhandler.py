import os
import numpy as np
from morphx.data.chunkhandler import ChunkHandler
from morphx.classes.pointcloud import PointCloud


def test_chunkhandler_sanity():
    wd = os.path.abspath(os.path.dirname(__file__) + '/../example_data/') + '/'
    radius = 20000
    npoints = 5000
    cl = ChunkHandler(wd, radius, npoints)

    for idx in range(len(cl)):
        sample = cl[idx]
        assert len(sample.vertices) == npoints
        assert len(sample.labels) == npoints

    cl.set_specific_mode(True)
    size = cl.get_hybrid_length('example_cell')
    for idx in range(size):
        sample = cl[('example_cell', idx)]
        assert len(sample.vertices) == npoints
        assert len(sample.labels) == npoints


def test_prediction_mapping():
    wd = os.path.abspath(os.path.dirname(__file__) + '/../example_data/') + '/'
    radius = 20000
    npoints = 5000
    cl = ChunkHandler(wd, radius, npoints, specific=True)

    sample = cl[('example_cell', 0)]
    pred_labels = np.arange(len(sample.vertices))
    pred_cloud = PointCloud(sample.vertices, pred_labels)




if __name__ == '__main__':
    test_chunkloader_sanity()
import os
import pytest
import numpy as np
from morphx.classes.pointcloud import PointCloud
from morphx.data.chunkhandler import ChunkHandler
from morphx.postprocessing.mapping import PredictionMapper


@pytest.mark.skip(reason="WIP")
def test_prediction_sanity():
    wd = os.path.abspath(os.path.dirname(__file__) + '/../example_data/') + '/'
    radius = 20000
    npoints = 5000
    cl = ChunkHandler(wd, radius, npoints)
    pm = PredictionMapper(wd, wd + 'predicted/', radius)

    cl.set_specific_mode(True)
    size = cl.get_hybrid_length('example_cell')
    for idx in range(size):
        sample = cl[('example_cell', idx)]
        labels = np.ones(len(sample.labels))*idx
        pred_cloud = PointCloud(sample.vertices, labels)
        pm.map_predictions(pred_cloud, 'example_cell', idx)
    pm.save_prediction()


if __name__ == '__main__':
    test_prediction_sanity()

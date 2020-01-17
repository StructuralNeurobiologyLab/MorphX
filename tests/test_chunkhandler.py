import os
import numpy as np
from morphx.data.chunkhandler import ChunkHandler
from morphx.processing import clouds


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


def test_transformations():
    wd = os.path.abspath(os.path.dirname(__file__) + '/../example_data/') + '/'
    radius = 20000
    npoints = 5000
    transforms = clouds.Compose([clouds.Normalization(radius), clouds.Center()])
    cl = ChunkHandler(wd, radius, npoints, transform=transforms)

    for idx in range(len(cl)):
        sample = cl[idx]
        assert len(sample.vertices) == npoints
        assert len(sample.labels) == npoints
        assert np.all(np.round(np.mean(sample.vertices, axis=0)) == np.array([0., 0., 0.]))


if __name__ == '__main__':
    test_chunkhandler_sanity()

import numpy as np
from morphx.processing import objects
from morphx.classes.hybridcloud import HybridCloud


def test_label_components():
    """
              3
              |
    1 -- 1 -- 2 -- 1 -- 2 -- 2 -- 3 -- 3 -- 1
    """
    nodes = np.random.random((10, 3))
    edges = np.array([(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)])
    node_labels = np.array([1, 1, 2, 3, 1, 2, 2, 3, 3, 1])
    hc = HybridCloud(nodes=nodes, edges=edges, node_labels=node_labels)
    reduced = objects.label_components(hc, 0)
    expected = {0: [0, 1], 1: [2], 2: [3], 3: [4], 4: [5, 6], 5: [7, 8], 6: [9]}
    for key in reduced:
        assert key in expected
        assert reduced[key] == expected[key]


if __name__ == '__main__':
    test_label_components()

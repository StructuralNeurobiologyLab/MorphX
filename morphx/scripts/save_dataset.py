import os
import numpy as np
from morphx.data.CloudSet import TorchClouds
from morphx.processing import clouds


if __name__ == '__main__':
    train_transform = clouds.Compose([clouds.RandomRotate(), clouds.RandomVariation(limits=(-50, 50)),
                                      clouds.Center()])

    epoch_size = 10

    data_path = os.path.expanduser('~/gt/gt_results/')
    data = TorchClouds(data_path, 10, 10000, transform=train_transform, epoch_size=epoch_size, pointcloud=True)

    for i in range(epoch_size):
        example, cloud = data[0]
        clouds.save_cloud(cloud, data_path + 'visualization', 'cloud_{}'.format(i))

        vertices = cloud.vertices
        centroid = np.mean(vertices, axis=0)
        print('{}: '.format(i) + str(centroid))

import os
from morphx.processing import clouds, visualize
from morphx.data.cloudset import CloudSet

train_transform = clouds.Identity()

data_path = os.path.expanduser('~/')
data = CloudSet(data_path, 40000, 1000, transform=train_transform)

samples = []
for i in range(10):
    samples.append(data[0])
    visualize.visualize_clouds(samples, capture=True, path='/home/john/test/it_{}.png'.format(i))


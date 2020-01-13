import os
import glob
from morphx.processing import ensembles, clouds
from morphx.classes.hybridcloud import HybridCloud


def create_ads(input_path: str, output_path: str):
    files = glob.glob(input_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for file in files:
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1]+1:-4]
        hc = clouds.load_cloud(file)
        hc = clouds.map_labels(hc, ['bouton', 'terminal'], 'axon')
        hc = clouds.map_labels(hc, ['neck', 'head'], 'dendrite')
        clouds.save_cloud(hc, output_path, name=name)


if __name__ == '__main__':
    create_ads('/u/jklimesch/gt/gt_poisson/', '/u/jklimesch/gt/gt_poisson/ads/')

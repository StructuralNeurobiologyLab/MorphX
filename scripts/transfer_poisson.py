import os
import glob
from morphx.processing import ensembles, clouds
from morphx.classes.hybridcloud import HybridCloud


def transfer_poisson(poisson_path: str, target_path: str, output_path: str):
    files = glob.glob(target_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for file in files:
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1]+1:-4]

        hc = clouds.load_cloud(poisson_path + name + '.pkl')
        ce = ensembles.load_ensemble(file)
        cell = ce.get_cloud('cell')

        encoding = {'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6}
        cell = HybridCloud(cell.nodes, cell.edges, hc.vertices, labels=hc.labels, encoding=encoding)
        ce.set_cloud(cell, 'cell')
        ensembles.save_ensemble(ce, output_path, name=name + '_c')


if __name__ == '__main__':
    transfer_poisson('/u/jklimesch/gt/gt_poisson/',
                     '/u/jklimesch/gt/gt_ensembles/raw/',
                     '/u/jklimesch/gt/gt_ensembles/')

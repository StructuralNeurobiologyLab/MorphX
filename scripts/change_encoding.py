import os
import glob
from morphx.processing import ensembles, clouds
from morphx.classes.hybridcloud import HybridCloud


def change_ensembles(input_path: str, output_path: str):
    files = glob.glob(input_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for file in files:
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1]+1:-4]
        ce = ensembles.load_ensemble(file)
        cell = ce.get_cloud('cell')
        encoding = {'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6}
        cell = HybridCloud(cell.nodes, cell.edges, cell.vertices, labels=cell.labels, encoding=encoding)
        ce.set_cloud(cell, 'cell')
        ensembles.save_ensemble(ce, output_path, name=name + '_c')


def change_hybrids(input_path: str, output_path: str):
    files = glob.glob(input_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for file in files:
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1]+1:-4]
        hc = clouds.load_cloud(file)
        encoding = {'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6}
        hc = HybridCloud(hc.nodes, hc.edges, hc.vertices, labels=hc.labels, encoding=encoding)
        clouds.save_cloud(hc, output_path, name=name + '_c')


if __name__ == '__main__':
    # change_ensembles('/u/jklimesch/gt/gt_ensembles/batch1/', '/u/jklimesch/gt/gt_ensembles/')
    change_hybrids('/u/jklimesch/gt/gt_poisson/wrong_encoding/', '/u/jklimesch/gt/gt_poisson/')

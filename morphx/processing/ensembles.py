# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import pickle as pkl
from typing import Optional
from morphx.classes.cloudensemble import CloudEnsemble


# -------------------------------------- ENSEMBLE I/O ------------------------------------------- #

# TODO: add saving in simple mode
def load_ensemble(path: str) -> Optional[CloudEnsemble]:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print("File with name: {} was not found at this location.".format(path))

    with open(path, 'rb') as f:
        obj = pkl.load(f)
    f.close()

    if isinstance(obj, CloudEnsemble):
        return obj
    else:
        return None


def save_ensemble(ensemble: CloudEnsemble, folder: str, name: str) -> int:
    full_path = os.path.join(folder, name + '.pkl')
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(full_path, 'wb') as f:
            pkl.dump(ensemble, f)
        f.close()
    except FileNotFoundError:
        print("Saving was not successful as given path is not valid.")
        return 1
    return 0

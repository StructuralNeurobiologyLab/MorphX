# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch, Philipp Schubert

from tqdm import tqdm
import numpy as np
import numba as nb
import warnings
from typing import Optional
from morphx.classes.hybridcloud import HybridCloud


class HybridMesh(HybridCloud):
    """ Class which represents a skeleton in form of a graph structure and a mesh which surrounds this skeleton. """

    def __init__(self,
                 faces: np.ndarray = None,
                 normals: Optional[np.ndarray] = None,
                 faces2node: dict = None,
                 *args, **kwargs):
        """
        Args:
            faces: The faces of the mesh as array of the respective vertices with shape (n, 3).
            normals: The normal vectors of the mesh.
            kwargs: See :py:class:`~morphx.classes.hybridcloud.HybridCloud`.
        """
        super().__init__(*args, **kwargs)

        if faces is None:
            faces = np.zeros((0, 3))
        if faces.shape[1] != 3:
            raise ValueError("Faces must have shape (N, 3).")
        self._faces = faces

        if normals is None:
            normals = np.zeros((0, 3))
        if len(normals) != 0 and len(normals) != len(faces):
            raise ValueError("Normals array must have same length as faces array.")
        self._normals = normals.reshape(len(normals), 3).astype(int)

        self._faces2node = faces2node

    @property
    def faces(self) -> np.ndarray:
        return self._faces

    @property
    def normals(self) -> np.ndarray:
        return self._normals

    @property
    def faces2node(self) -> Optional[dict]:
        # TODO: slow, optimization required.
        """ Creates python dict with indices of ``:py:attr:~nodes`` as
        keys and lists of face indices associated with the nodes as values.
        All faces which contain at least ONE "node"-vertex are associated with
        this node.

        Returns:
            Python dict with mapping information.
        """
        if len(self.nodes) == 0:
            return None
        if self._faces2node is None:
            self._faces2node = dict()
            for node_ix, vert_ixs in self.verts2node.items():
                if len(vert_ixs) == 0:
                    self._faces2node[node_ix] = []
                    continue
                # TODO: Depreciated use of set with numba => change when numba
                #  has published typed_set
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    new_faces = any_in_1d_nb(self.faces, set(vert_ixs))
                new_faces = np.nonzero(new_faces)[0].tolist()
                self._faces2node[node_ix] = new_faces
        return self._faces2node

    def get_attr_dict(self):
        attr_dict = {'faces': self._faces, 'normals': self._normals, 'faces2node': self.faces2node}
        attr_dict.update(super().get_attr_dict())
        return attr_dict


@nb.njit(parallel=True)
def any_in_1d_nb(matrix, indices):
    out = np.empty(matrix.shape[0], dtype=nb.boolean)
    for i in nb.prange(matrix.shape[0]):
        if (matrix[i, 0] in indices) or (matrix[i, 1] in indices) or (matrix[i, 2] in indices):
            out[i] = True
        else:
            out[i] = False
    return out

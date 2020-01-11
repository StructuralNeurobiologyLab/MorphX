# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
from typing import Dict, Optional
from morphx.classes.pointcloud import PointCloud


class CloudEnsemble(object):
    """
    Class which represents a collection of PointCloud objects.
    """

    def __init__(self, clouds: Dict[str, PointCloud]):
        """
        Args:
            clouds: Dict with cloud names as keys and PointCloud objects as Values.
        """
        self._clouds = clouds
        if len(self._clouds) == 0:
            raise ValueError("CloudEnsemble must contain at least one cloud.")

    @property
    def clouds(self):
        return self._clouds

    def get_cloud(self, cloud_name: str) -> Optional[PointCloud]:
        try:
            return self._clouds[cloud_name]
        except ValueError:
            return None

    def set_cloud(self, cloud: PointCloud, cloud_name: str):
        self._clouds[cloud_name] = cloud

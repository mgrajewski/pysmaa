"""
@author: Matthias Grajewski, FH Aachen University of Applied Sciences
Luis Hasenauer, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""

import numpy as np
from numpy.typing import NDArray
from src.mcda_methods.McdaMethod import McdaMethod


class SawMinMax(McdaMethod):
    def __init__(self, benefit: NDArray[bool]):
        """
        Parameters
        ----------
        benefit: 1D-numpy array (bool)
            if true, the i-th criterion is associated to a benefit, if false, to costs.
        """
        self.benefit = benefit

    def p_from_perfmat(self, perfmat: NDArray[float]) -> NDArray[float]:
        """
        This function computes from the raw performance matrix perfmat the SAW-matrix P. Here, benefit criteria are
        scaled by dividing through the column maximum and cost criteria are inverted and multiplied with the column
        minimum.
        We assume that perfmat is valid, i.e. does not contain duplicate rows. You can check this using the separate
        function check_P.

        Parameters
        ----------
        perfmat : 2D-numpy-array
            raw performance matrix

        Returns
        -------
        p: 2D-numpy-array
            the normalized SAW-matrix
        """
        p = np.zeros(perfmat.shape, dtype=float)
        if np.any(self.benefit):
            p[:, self.benefit] = perfmat[:, self.benefit] / np.max(perfmat[:, self.benefit], axis=0)
        if np.any(~self.benefit):
            p[:, ~self.benefit] = np.min(perfmat[:, ~self.benefit], axis=0) / perfmat[:, ~self.benefit]
        return p

    def get_values_of_alts(self, perfmat: NDArray[float], weights: NDArray[float]) -> NDArray[float]:
        """
        This function computes from the raw performance matrix perfmat ant the weights the values of the alternatives.
        Non-benefit entries in the performance matrix are set to their respective negative values.
        We assume that perfmat is valid, i.e. does not contain duplicate rows. You can check this using the separate
        function check_P.

        Parameters
        ----------
        perfmat : 2D-numpy-array
            raw performance matrix

        weights : 1D-numpy-array
            weights of the criteria

        Returns
        -------
        values: 1D-numpy-array
            the values of the alternatives
        """

        p = np.zeros(perfmat.shape, dtype=float)
        if np.any(self.benefit):
            p[:, self.benefit] = perfmat[:, self.benefit] / np.max(perfmat[:, self.benefit], axis=0)
        if np.any(~self.benefit):
            p[:, ~self.benefit] = np.min(perfmat[:, ~self.benefit], axis=0) / perfmat[:, ~self.benefit]

        return p @ weights

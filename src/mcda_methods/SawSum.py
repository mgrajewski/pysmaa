"""
@author: Luis Hasenauer, FH Aachen University of Applied Sciences
Matthias Grajewski, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""

import numpy as np
from numpy.typing import NDArray
from src.mcda_methods.McdaMethod import McdaMethod


class SawSum(McdaMethod):
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
        This function computes from the raw performance matrix perfmat the SAW-matrix P. Here, this restricts to
        scaling the matrix columns such that the column-sum of the absolute values is 1. Non-benefit entries are set to
        their respective negative values.
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
        perfmat[:, ~self.benefit] = -perfmat[:, ~self.benefit]
        return perfmat / np.sum(np.abs(perfmat), axis=0)

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

        perfmat[:, ~self.benefit] = -perfmat[:, ~self.benefit]
        perfmat = perfmat / np.sum(np.abs(perfmat), axis=0)
        return perfmat @ weights

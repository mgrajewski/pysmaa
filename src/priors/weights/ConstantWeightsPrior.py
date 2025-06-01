"""
@author: Luis Hasenauer (main contributor), FH Aachen University of Applied Sciences
Matthias Grajewski, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""

import numpy as np

from src.weights_from_rankings import get_q
from numpy.typing import NDArray


class ConstantWeightsPrior:
    """
    A class to represent a fixed weight as prior, i.e. only the performance matrix and the parameters of the MCDA model
    are considered uncertain.
    """

    def __init__(self, weight: NDArray[float]) -> None:
        """

        Parameters
        ----------
        weight : 1D-numpy-array
            the fixed weight prescribed as prior
        """
        self.weight = weight

        self.Q = get_q(weight.size)[:, :-1]
        self.cog = np.ones(weight.size) / weight.size

        # Transformed weights have one dimensions less than the "real" ones.
        self.weight_trans = (weight - self.cog) @ self.Q

    def log_density(self, w: NDArray[float]) -> float:
        """
        Computes the logarithm of the probability density function of w.

        Parameters
        ----------
        w : 1D-numpy-array
            the weight vector

        Returns
        -------
        float
            the logarithm of the density function
        """
        if np.linalg.norm(w - self.weight, 2) < 1e-12:
            return 0
        else:
            return -np.Inf

    def sample(self, n_samples: int) -> NDArray[float]:
        """
        This function computes n_samples samples of the weights. As the weights are considered constant, this is
        n_samples time the weight itself. It does not make too much sense to use this function anyway, it is written
        more or less for compatibility reasons.

        Parameters
        ----------
        n_samples : number of samples

        Returns
        -------
        samples: 2d-numpy array
            sampled admissible weights
        """
        if n_samples < 0:
            raise NameError('The number of samples must be at least 0.')

        return np.tile(self.weight, (n_samples, 1))

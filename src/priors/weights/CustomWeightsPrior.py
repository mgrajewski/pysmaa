"""
@author: Matthias Grajewski, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""

from numpy.typing import NDArray
from typing import Callable


class CustomWeightsPrior:
    """
    This class is to represent a custom prior distribution of the weights on the standard simplex. It is merely a
    wrapper for a user-provided function for computing the log density and for sampling according to the prior
    distribution.
    """

    def __init__(self, log_density_by_user: Callable, sample_by_user: Callable) -> None:
        """

        Parameters
        ----------
        log_density_by_user : custom, user-provided function for computing the log density
        sample_by_user : custom, user-provided function for sampling weights according to the prior distribution
        """
        self.log_density = log_density_by_user
        self.sample = sample_by_user

    def log_density(self, w: NDArray[float]) -> float:
        """
        Computes the logarithm of the prior probability density function at w.

        Parameters
        ----------
        w : 1D-numpy-array
            the weight vector

        Returns
        -------
        float
            the logarithm of the density function
        """
        return self.log_density(w)

    def sample(self, n_samples: int) -> NDArray[float]:
        """
        This function computes n_samples samples from the custom distribution.

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

        samples = self.sample(n_samples)

        return samples[:min(n_samples, samples.shape[0]), :]

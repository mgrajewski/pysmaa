"""
@author: Luis Hasenauer (main contributor), FH Aachen University of Applied Sciences
Matthias Grajewski, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""

import numpy as np
from numpy.typing import NDArray


class UniformWeightsPrior:
    """
    A class to represent a uniform prior distribution of the weight vector on the standard simplex.
    """

    def __init__(self, dim: int) -> None:

        if dim <= 0:
            raise NameError('The dimension dim must be positive.')

        self.dim = dim

    def log_density(self, w: NDArray[float]) -> float:
        """
        Computes the logarithm of the (non-normalized) probability density function of a uniform model. The parameter
        w is actually not necessary, but is needed to keep the function signature aligned with the log_density
        functions for other distributions.

        Parameters
        ----------
        w : 1D-numpy-array
            the weight vector

        Returns
        -------
        float
            The (non-normalized) logarithm of the prior density
        """
        return 0

    def sample(self, n_samples: int, seed_for_rng: None|int=None) -> NDArray[float]:
        """
        This function computes n_samples samples from the uniform distribution on the simplex.

        Parameters
        ----------
        n_samples : int
            number of samples
        seed_for_rng : integer or None (default value)
            Seed for the random number generator in numpy. By default, seed=None, which means that numpy does not use a
            specific seed. We recommend setting a seed for debugging only.

        Returns
        -------
        samples: 2d-numpy array
            sampled admissible weights
        """

        if n_samples < 0:
            raise NameError('The number of samples must be at least 0.')

        # Set up the random number generator. Here, we use the standard number generator from numpy.
        rng = np.random.default_rng(seed=seed_for_rng)

        samples = rng.exponential(size=(self.dim, n_samples))

        # normalize
        samples = samples / np.sum(samples, axis=0)

        return samples.T


def main():

    # small test example for sampling.
    dim = 3
    my_uniform_weights_prior = UniformWeightsPrior(dim)
    samples = my_uniform_weights_prior.sample(10, seed_for_rng=3)
    print(samples)

if __name__ == '__main__':
    main()

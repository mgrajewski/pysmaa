"""
@author: Luis Hasenauer, FH Aachen University of Applied Sciences
Matthias Grajewski, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""

import numpy as np
from numpy.typing import NDArray
import warnings


class FixedPerfMatPrior:
    """
    This class represents a prior distribution of the performance matrix if it is fixed.
    """
    def __init__(self, perfmat: NDArray[float]):
        """
        Parameters
        ----------
        perfmat : 2D-numpy-array
            the raw performance matrix
        """
        self.perfmat_mean = perfmat

    def sample(self, n_samples=100, seed_for_rng: None|int=None):
        """
        Parameters
        ----------
        n_samples : integer
            required number of samples of the performance matrix
        seed_for_rng : integer
            dummy variable. It is necessary to keep the function signature aligned to the corresponding sampling methods
            for uncertain performance matrices.

        Returns
        -------
        3D-numpy-array (float) with size (n_samples, n_alts, n_crit)
            samples of the (raw) performance matrix
        """

        if n_samples < 0:
            raise NameError('The number of samples must be at least 0.')

        # Issue a warning that sampling a fixed performance matrix makes little sense.
        warn_msg = 'Sampling of a fixed performance matrix occurred, which is inefficient.'
        warnings.warn(warn_msg)

        [n_alts, n_crit] = self.perfmat_mean.shape
        perfmat_samples = np.zeros((n_samples, n_alts, n_crit))

        # This can be implemented more efficiently using fancy numpy operations. However, it is even more efficient to
        # skip sampling a fixed matrix at all.
        for i in range(n_samples):
            perfmat_samples[i] = self.perfmat_mean

        return perfmat_samples


def main():
    # Small test example
    n_alts = 2
    n_crit = 3
    perfmat = np.ones((n_alts, n_crit), dtype=float)
    prior = FixedPerfMatPrior(perfmat)

    # sample 5 instances of the performance matrix
    samples_perfmat = prior.sample(5)
    print(samples_perfmat)


if __name__ == '__main__':
    main()

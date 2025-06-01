"""
@author: Luis Hasenauer (main contributor), FH Aachen University of Applied Sciences
Matthias Grajewski, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""

import numpy as np
import hopsy
import warnings

from src.weights_from_rankings import get_q
from numpy.typing import NDArray


class NormalWeightsPrior:
    """
    A class to represent a multivariate normal prior distribution. This class is wrapping hopsy.Gaussian to allow
    sampling of a degenerate distribution on the standard-simplex by transforming it to dimension -1.
    """

    def __init__(self, mean: NDArray[float], covariance: NDArray[float]) -> None:
        """
        Transforms the given mean and covariance and creates a hopsy.Gaussian object, which is necessary for MCMC
        sampling with hopsy using a Gaussian prior.
        Note: We do not use the hopsy.Gaussian object for sampling, as it does not provide a sampling method, but is
        intended to provide the log density in MCMC sampling. Instead, we rely on thy standard random number generator
        (rng) of numpy.
        However, hopsy requires this object if we want to use Gaussian prior in hopsy, so just removing it is no
        option.

        Parameters
        ----------
        mean : 1D-numpy-array
            the mean
        covariance : 2D-numpy-array
            the covariance matrix
        """
        self.mean = mean

        self.Q = get_q(mean.size)[:, :-1]
        self.cog = np.ones(mean.size) / mean.size

        # Transformed weights have one dimensions less than the "real" ones.
        mean_trans = (mean - self.cog) @ self.Q
        self.mean_trans = mean_trans

        covariance_trans = self.Q.T @ covariance @ self.Q
        self.covariance_trans = covariance_trans
        self.model = hopsy.Gaussian(mean_trans, covariance_trans)

    def log_density(self, w: NDArray[float]) -> float:
        """
        Computes the logarithm of the probability density function of a multivariate normal distribution at w.

        Parameters
        ----------
        w : 1D-numpy-array
            the weight vector

        Returns
        -------
        float
            the logarithm of the density function
        """
        return self.model.log_density((w - self.cog) @ self.Q)

    def sample(self, n_samples: int, seed_for_rng: None|int=None) -> NDArray[float]:
        """
        This function computes n_samples samples from the (truncated) multivariate normal distribution using naive
        rejection sampling. The distribution is truncated as only admissible weights are sampled.

        Parameters
        ----------
        n_samples : int
            number of samples
        seed_for_rng : int or None (default value)
            Seed for the random number generator in numpy. By default, seed=None, which means that numpy does not use a
            specific seed. We recommend setting a seed for debugging only.

        Returns
        -------
        samples : 2d-numpy array
            sampled admissible weights
        """
        eps = -1e-12
        n_max_trials = 50

        if n_samples < 0:
            raise NameError('The number of samples must be at least 0.')

        i_trial = 0
        samples = np.zeros((0, self.mean_trans.shape[0]+1))

        # Set up the random number generator. Here, we use the standard number generator from numpy.
        rng = np.random.default_rng(seed=seed_for_rng)

        while samples.shape[0] < n_samples and i_trial < n_max_trials:

            # The covariance matrix is singular in real weights, but not in transformed weights. Therefore, we use
            # transformed weights but recompute real weights afterward.
            samples_new_trans = rng.multivariate_normal(self.mean_trans, self.covariance_trans, n_samples)

            samples_new = samples_new_trans@self.Q.T + self.cog

            contains_no_negatives = (samples_new > eps).all(axis=1)
            samples_new = samples_new[contains_no_negatives]

            samples = np.append(samples, samples_new, axis=0)
            i_trial += 1

        if i_trial == n_max_trials:
            warnings.warn('Maximum number of sampling runs reached. The majority of samples is discarded.')

        if samples.shape[0] == 0:
            raise NameError('Sampling according to the prior failed, not even a single sample was produced.')

        return samples[:min(n_samples, samples.shape[0]), :]


def main():

    # small test example for sampling.
    dim = 3
    mean = np.array([1.0,0,0], dtype=float)
    covariance = 0.1*np.eye(dim)
    my_normal_weights_prior = NormalWeightsPrior(mean, covariance)
    samples = my_normal_weights_prior.sample(10, 3)
    print(samples)

if __name__ == '__main__':
    main()

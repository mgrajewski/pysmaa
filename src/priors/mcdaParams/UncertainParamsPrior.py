"""
@author: Matthias Grajewski, FH Aachen University of Applied Sciences
Luis Hasenauer, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""

import numpy as np
from scipy.stats import uniform, norm
from numpy.typing import NDArray
from typing import Optional, Callable


class UncertainParamsPrior:
    """
    This class represents the prior distribution of the parameters of the MCDA model where every parameter can follow an
    individual distribution.
    """
    def __init__(self, params_mean: NDArray[float], param_types: NDArray[int], param_params: NDArray[float],
                 custom_density: Optional[Callable[[NDArray[float]], NDArray[float]]] = None,
                 custom_dist: Optional[Callable[[NDArray[float], NDArray[float], int], NDArray[float]]] = None):
        """
        Parameters
        ----------
        params_mean : 2D-numpy-array (float)
            The mean of the model parameters
        perfmat_types : 2D-numpy-array (int)
            Entry (i, j) of the matrix resembles the distribution of entry (i, j) of perfmat:
                0 - the entry is fixed to the mean
                1 - the entry follows a uniform distribution
                2 - the entry follows a normal distribution
                10 - the entry follows a custom distribution, determined in custom_function
        perfmat_params : 2D-numpy-array (float)
            Entry (i, j) of the matrix resembles the distribution parameter of entry (i, j) of perfmat:
                perfmat_types[i, j] == 0 - NaN
                perfmat_types[i, j] == 1 - the parameter resembles the maximum relative difference to the mean:
                    perfmat[i, j] ~ Unif(mean - param * mean, mean + param * mean)
                perfmat_types[i, j] == 2 - the parameter resembles the variance of the normal distribution
                perfmat_types[i, j] == 10 - NaN
        custom_density : callable[[1D-numpy-array], float]
            Function of form  1D-numpy-array -> float returning the density of the custom entries of perfmat at the point
            given as a parameter. The input is flattened in row order.
        """
        self.perfmat_mean = perfmat_mean
        self.perfmat_types = perfmat_types
        self.perfmat_params = perfmat_params
        self.custom_density = custom_density
        self.custom_dist = custom_dist

    def log_density(self, perfmat: NDArray[float]) -> float:
        """
        Parameters
        ----------
        perfmat : 2D-numpy-array
            the point where the density should be computed

        Returns
        -------
        float
            the logarithm of the density function of perfmat
        """
        if np.any(perfmat < 0):
            return -np.inf
        log_prob = 0
        # Uniform distributed entries
        uniform_indices = self.perfmat_types == 1
        if np.any(uniform_indices):
            lower_bounds = (1 - self.perfmat_params[uniform_indices]) * self.perfmat_mean[uniform_indices]
            widths = 2 * self.perfmat_params[uniform_indices] * self.perfmat_mean[uniform_indices]
            log_prob += np.sum(uniform.logpdf(perfmat[uniform_indices], lower_bounds, widths))
        # Normal distributed entries
        normal_indices = self.perfmat_types == 2
        if np.any(normal_indices):
            means = self.perfmat_mean[normal_indices]
            standard_deviations = np.sqrt(self.perfmat_params[normal_indices])
            log_prob += np.sum(norm.logpdf(perfmat[normal_indices], means, standard_deviations))
        # Custom distributed entries
        custom_indices = self.perfmat_types == 10
        if np.sum(custom_indices) >= 1:
            if self.custom_density is None:
                raise Exception("Some entries of the performance matrix are supposed to have a custom distribution, "
                                "but there is no custom sample function provided.")
            log_prob += np.log(self.custom_density(self.perfmat_mean[custom_indices]))
        return log_prob

    def sample(self, n_samples=100):

        [nalts, ncrit] = self.perfmat_mean.shape

        fixed_indices = (self.perfmat_types == 0)
        uniform_indices = (self.perfmat_types == 1)
        normal_indices = (self.perfmat_types == 2)
        custom_indices = (self.perfmat_types == 10)

        perfmat_samples = np.zeros((n_samples, nalts, ncrit))

        if np.sum(fixed_indices) > 1:
            perfmat_samples[:, fixed_indices] = self.perfmat_mean[fixed_indices]

        # uniformly distributed matrix entries
        if np.sum(uniform_indices) > 1:
            perfmat_samples[:, uniform_indices] = \
                np.random.uniform((1 - self.perfmat_params[uniform_indices]) * self.perfmat_mean[uniform_indices],
                                  (1 + self.perfmat_params[uniform_indices]) * self.perfmat_mean[uniform_indices],
                                  size=(n_samples, self.perfmat_params[uniform_indices].shape[0]))

        # normally distributed matrix entries
        if np.sum(normal_indices) > 1:
            perfmat_samples[:, normal_indices] = \
                np.random.normal(self.perfmat_mean[normal_indices], self.perfmat_params[normal_indices], size=n_samples)

        # matrix entries with custom distribution
        if np.sum(custom_indices) >= 1:
            if self.custom_dist is None:
                raise Exception("Some entries of the performance matrix are supposed to have a custom distribution, "
                                "but there is no custom sample function provided.")

            perfmat_samples[:, custom_indices] = self.custom_dist(self.perfmat_mean[custom_indices],
                                                                 self.perfmat_params[custom_indices], n_samples)

        # no negative entries allowed in performance matrix
        perfmat_samples = np.clip(perfmat_samples, a_min=0, a_max=None)
        return perfmat_samples


def main():
    # Small test example
    n_alts = 2
    n_crits = 3
    perfmat_mean = np.ones((n_alts, n_crits), dtype=float)
    perfmat_types = np.array([[1., 1, 1], [1, 1, 1]], dtype=int)
    perfmat_params = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], dtype=float)
    prior = UncertainPerfMatPrior(perfmat_mean, perfmat_types, perfmat_params)
    print(np.exp(prior.log_density(perfmat_mean)))

    # sample 5 instances of the performance matrix
    samples_perfmat = prior.sample(5)
    print(samples_perfmat)


if __name__ == '__main__':
    main()

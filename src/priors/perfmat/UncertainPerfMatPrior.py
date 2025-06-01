"""
@author: Matthias Grajewski, FH Aachen University of Applied Sciences
Luis Hasenauer, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""

import numpy as np
from scipy.stats import uniform, norm
from numpy.typing import NDArray
from typing import Optional, Callable


class UncertainPerfMatPrior:
    """
    This class represents the prior distribution of the performance matrix perfmat where every entry can follow an
    individual distribution.
    """

    def __init__(self, perfmat_mean: NDArray[float], perfmat_types: NDArray[int], perfmat_params: NDArray[float],
                 custom_density: Optional[Callable[[NDArray[float]], NDArray[float]]] = None,
                 custom_dist: Optional[Callable[[NDArray[float], NDArray[float], int], NDArray[float]]] = None):
        """
        Parameters
        ----------
        perfmat_mean : 2D-numpy-array (float)
            The mean of the raw performance matrix perfmat
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
                    perfmat[i, j] ~ Uniform(mean - param * mean, mean + param * mean)
                perfmat_types[i, j] == 2 - the parameter resembles the variance of the normal distribution
                perfmat_types[i, j] == 10 - NaN
        custom_density : callable[[1D-numpy-array], float]
            Function of form  1D-numpy-array -> float returning the density of the custom entries of perfmat at the
            point given as a parameter. The input is flattened in row order.
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
        perfmat : 2D-numpy-array (float)
            the instance of the performance matrix for which the logarithm of the density function is to be computed

        Returns
        -------
        float
            the logarithm of the density function at perfmat
        """
        if np.any(perfmat < 0):
            return -np.inf
        log_prob = 0

        # Uniformly distributed entries
        uniform_indices = self.perfmat_types == 1
        normal_indices = self.perfmat_types == 2
        custom_indices = self.perfmat_types == 10

        uniform_indices_rel = np.bitwise_and(uniform_indices, self.perfmat_params >= 0)
        uniform_indices_abs = np.bitwise_and(uniform_indices, self.perfmat_params < 0)

        if np.any(uniform_indices_rel):
            # a positive parameter indicates relative uncertainty, a negative one absolute uncertainty
            lower_bounds = (1 - self.perfmat_params[uniform_indices_rel]) * self.perfmat_mean[uniform_indices_rel]
            widths = 2 * self.perfmat_params[uniform_indices_rel] * self.perfmat_mean[uniform_indices_rel]
            log_prob += np.sum(uniform.logpdf(perfmat[uniform_indices_rel], lower_bounds, widths))

        if np.any(uniform_indices_abs):
            # a positive parameter indicates relative uncertainty, a negative one absolute uncertainty

            lower_bounds = np.clip(self.perfmat_mean[uniform_indices_abs] + self.perfmat_params[uniform_indices_abs],
                                   a_min=0.0, a_max=None)
            widths = self.perfmat_mean[uniform_indices_abs] - self.perfmat_params[uniform_indices_abs] - lower_bounds
            log_prob += np.sum(uniform.logpdf(perfmat[uniform_indices_abs], lower_bounds, widths))

        # Normally distributed entries
        if np.any(normal_indices):
            means = self.perfmat_mean[normal_indices]
            standard_deviations = np.sqrt(self.perfmat_params[normal_indices])
            log_prob += np.sum(norm.logpdf(perfmat[normal_indices], means, standard_deviations))

        # Custom distributed entries
        if np.any(custom_indices):
            if self.custom_density is None:
                raise Exception("Some entries of the performance matrix are supposed to have a custom distribution, "
                                "but there is no custom sample function provided.")
            log_prob += np.log(self.custom_density(self.perfmat_mean[custom_indices]))
        return log_prob

    def sample(self, n_samples:int=100, seed_for_rng: None|int=None):
        """
        Parameters
        ----------
        n_samples : integer
            required number of samples of the performance matrix
        seed_for_rng : integer or None (default value)
            Seed for the random number generator in numpy. By default, seed=None, which means that numpy does not use a
            specific seed. We recommend setting a seed for debugging only.

        Returns
        -------
        3D-numpy-array (float) with size (n_samples, n_alts, n_crit)
            samples of the (raw) performance matrix
        """
        if n_samples < 0:
            raise NameError('The number of samples must be at least 0.')

        n_max_trials = 3

        # Set up the random number generator. Here, we use the standard number generator from numpy.
        rng = np.random.default_rng(seed=seed_for_rng)

        [n_alts, n_crit] = self.perfmat_mean.shape

        fixed_indices = (self.perfmat_types == 0)
        uniform_indices = (self.perfmat_types == 1)
        normal_indices = (self.perfmat_types == 2)
        custom_indices = (self.perfmat_types == 10)

        uniform_indices_rel = np.bitwise_and(uniform_indices, self.perfmat_params >= 0)
        uniform_indices_abs = np.bitwise_and(uniform_indices, self.perfmat_params < 0)

        perfmat_samples = np.zeros((n_samples, n_alts, n_crit))

        if np.any(fixed_indices):
            perfmat_samples[:, fixed_indices] = self.perfmat_mean[fixed_indices]

        # uniformly distributed matrix entries
        if np.any(uniform_indices_rel):
            perfmat_samples[:, uniform_indices_rel] = \
                rng.uniform((1 - self.perfmat_params[uniform_indices_rel]) * self.perfmat_mean[uniform_indices_rel],
                            (1 + self.perfmat_params[uniform_indices_rel]) * self.perfmat_mean[uniform_indices_rel],
                            size=(n_samples, self.perfmat_params[uniform_indices_rel].shape[0]))

        if np.any(uniform_indices_abs):
            # a positive parameter indicates relative uncertainty, a negative one absolute uncertainty

            lower_bounds = np.clip(self.perfmat_mean[uniform_indices_abs] + self.perfmat_params[uniform_indices_abs],
                                   a_min=0.0, a_max=None)
            perfmat_samples[:, uniform_indices_abs] = \
                rng.uniform(lower_bounds,
                            self.perfmat_mean[uniform_indices_abs] - self.perfmat_params[uniform_indices_abs],
                            size=(n_samples, self.perfmat_params[uniform_indices_abs].shape[0]))

        # normally distributed matrix entries
        if np.any(normal_indices):
            continue_sampling = True
            n_trials = 0
            samples_normal = np.zeros((0, self.perfmat_params[normal_indices].shape[0]))

            while continue_sampling:
                n_trials += 1

                aux_arr = rng.normal(self.perfmat_mean[normal_indices], self.perfmat_params[normal_indices],
                                     size=(n_samples, self.perfmat_params[normal_indices].shape[0]))

                # We use, in fact, truncated normally distributed entries, such that we discard all samples where negative
                # values occur.
                aux_arr = aux_arr[np.all(aux_arr >= 0, axis=1)]
                samples_normal = np.append(samples_normal, aux_arr, axis=0)

                if (samples_normal.shape[0] >= n_samples) or n_trials >= n_max_trials:
                    continue_sampling = False

            perfmat_samples[:samples_normal.shape[0], normal_indices] = samples_normal[0:n_samples]

            # Obviously, rejection sampling is inefficient, as the majority of samples is discarded. We now exploit
            # the independence of the matrix entries and sample every entry separately.
            if n_trials == n_max_trials and samples_normal.shape[0] < n_samples:

                n_max_trials = 10

                n_samples_left = n_samples - samples_normal.shape[0]

                explicit_normal_indices = np.array(np.where(normal_indices)).T

                for i_entry in explicit_normal_indices:

                    continue_sampling = True
                    n_trials = 0

                    samples_normal_2 = np.zeros(0)

                    while continue_sampling:
                        n_trials += 1
                        aux_arr = rng.normal(self.perfmat_mean[i_entry[0], i_entry[1]],
                                             self.perfmat_params[i_entry[0], i_entry[1]], size=n_samples_left)

                        aux_arr = aux_arr[aux_arr >= 0]
                        samples_normal_2 = np.append(samples_normal_2, aux_arr, axis=0)

                        if (samples_normal_2.shape[0] >= n_samples_left) or n_trials >= n_max_trials:
                            continue_sampling = False

                    if n_trials == n_max_trials and samples_normal_2.shape[0] < n_samples:
                        raise Exception('Sampling of normally distributed entries of the performance matrix failed.')

                    perfmat_samples[samples_normal.shape[0]:, i_entry[0], i_entry[1]] = \
                        samples_normal_2[0:n_samples_left]

        # matrix entries with custom distribution
        if np.any(custom_indices):
            if self.custom_dist is None:
                raise Exception("Some entries of the performance matrix are supposed to have a custom distribution, "
                                "but there is no custom sample function provided.")

            perfmat_samples[:, custom_indices] = self.custom_dist(self.perfmat_mean[custom_indices],
                                                                  self.perfmat_params[custom_indices], n_samples)

        return perfmat_samples


def main():
    # Small test example
    n_alts = 2
    n_crit = 3
    perfmat_mean = np.ones((n_alts, n_crit), dtype=float)
    perfmat_types = np.array([[1., 1, 1], [1, 1, 1]], dtype=int)
    perfmat_params = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], dtype=float)
    prior = UncertainPerfMatPrior(perfmat_mean, perfmat_types, perfmat_params)
    print(np.exp(prior.log_density(perfmat_mean)))

    # sample 5 instances of the performance matrix
    samples_perfmat = prior.sample(5, seed_for_rng=3)
    print(samples_perfmat)


if __name__ == '__main__':
    main()

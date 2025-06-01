"""
@author: Luis Hasenauer, FH Aachen University of Applied Sciences
Matthias Grajewski, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""
from src.priors.perfmat.UncertainPerfMatPrior import UncertainPerfMatPrior
from src.priors.perfmat.FixedPerfMatPrior import FixedPerfMatPrior
from numpy.typing import NDArray
from typing import Union
from src.priors.weights.UniformWeightsPrior import UniformWeightsPrior
from src.priors.weights.NormalWeightsPrior import NormalWeightsPrior
from src.priors.weights.CustomWeightsPrior import CustomWeightsPrior


class Prior:
    """
    This class combines the prior distributions of the weights and performance matrix.
    """

    def __init__(self, weights_prior: Union[UniformWeightsPrior, NormalWeightsPrior, CustomWeightsPrior],
                 perfmat_prior: Union[UncertainPerfMatPrior, FixedPerfMatPrior]) -> None:
        """
        Parameters
        ----------
        weights_prior : Union[UniformWeightsPrior, NormalWeightsPrior, CustomWeightsPrior]
            the prior distribution of the weight vector
        perfmat_prior : Union[UncertainPerfMatPrior, FixedPerfMatPrior]
            the prior distribution of the performance matrix perfmat
        """
        self.weights_prior = weights_prior
        self.perfmat_prior = perfmat_prior
        self.n_crits = self.perfmat_prior.perfmat_mean.shape[1]

    def log_density(self, theta: NDArray[float]) -> float:
        """
        Computes the product of the densities of the weight vector and performance matrix perfmat.

        Parameters
        ----------
        theta : 1D-numpy-array
            the parameter vector of form (w, flat(perfmat))

        Returns
        -------
        float
            the logarithm of the prior density at theta
        """
        w = theta[:self.n_crits]
        perfmat = theta[self.n_crits:].reshape(-1, self.n_crits)
        return self.weights_prior.log_density(w) + self.perfmat_prior.log_density(perfmat)

    def sample(self, n_samples, seed_for_rng: None|int=None):
        """
        Samples the weights and performance matrix perfmat according to the prior distribution.

        Parameters
        ----------
        n_samples: integer
            the number of samples
        seed_for_rng : integer or None (default value)
            Seed for the random number generator in numpy. By default, seed=None, which means that numpy does not use a
            specific seed. We recommend setting a seed for debugging only.

        Returns
        -------
        list
            The first entry contains the samples of the weights, the second one the samples of perfmat
        """
        if seed_for_rng is None:
            samples_weights = self.weights_prior.sample(n_samples, seed_for_rng)
            samples_perfmat = self.perfmat_prior.sample(n_samples, seed_for_rng)
        else:
            # Ensure that the rng for weights and performance matrix are initialised with different seeds
            samples_weights = self.weights_prior.sample(n_samples, seed_for_rng)
            samples_perfmat = self.perfmat_prior.sample(n_samples, 2*seed_for_rng)


        return [samples_weights, samples_perfmat]

"""
@author: Matthias Grajewski, FH Aachen University of Applied Sciences
Luis Hasenauer, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""
import numpy as np
from numpy.typing import NDArray
from src.mcda_methods.McdaMethod import McdaMethod


class Likelihood:
    def __init__(self, n_alts: int, n_crits: int, ranking: NDArray[int], mcda_method: McdaMethod):
        """
        Parameters
        ----------
        n_alts : int
            the number of alternatives
        n_crits : int
            the number of criteria
        ranking : 1D-numpy-array
            ranking of alternatives (0-based)
        mcda_method : McdaMethod
            given MCDA-method
        """
        self.n_alts = n_alts
        self.n_crits = n_crits
        self.ranking = ranking
        self.mcda_method = mcda_method

    def log_density(self, theta: NDArray[float]) -> float:
        """
        Splits the parameter vector into the weight vector and the performance matrix perfmat to compute the logarithm
        of the likelihood function.

        Parameters
        ----------
        theta : 1D-numpy-array
            the parameter vector of form (w, flat(perfmat))

        Returns
        -------
        float
            the logarithm of the likelihood function at point theta
        """
        w = theta[:self.n_crits]
        ptilde = theta[self.n_crits:].reshape(self.n_alts, self.n_crits)
        return self._log_density(w, ptilde)

    def _log_density(self, w: NDArray[float], perfmat: NDArray[float]) -> float:
        """
        This private function computes the logarithm of the likelihood function for a given weighting vector w and a
        given performance matrix perfmat. The likelihood is one if this combination of w and perfmat results in the
        given ranking using the given MCDA-method, else zero.

        Parameters
        ----------
        w : 1D-numpy-array
            the weighting vector
        perfmat : 2D-numpy-array
            the performance matrix

        Returns
        -------
        float
            the logarithm of the likelihood function at (w, perfmat)
        """
        if self.ranking.size == 0:
            return 0.0

        performances = self.mcda_method.get_values_of_alts(perfmat, w)

        ranking = np.argsort(-performances)[:self.ranking.size]

        # For given samples of w, P and gamma (the parameters of the MCDA model), the likelihood is either one or zero.
        # Therefore, we return either log(1) = 0 or log(0) = -inf.
        if np.all(ranking == self.ranking):
            return 0.0
        else:
            return -np.inf

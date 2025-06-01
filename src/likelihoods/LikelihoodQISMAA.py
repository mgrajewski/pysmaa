"""
@author: Matthias Grajewski, FH Aachen University of Applied Sciences
Luis Hasenauer, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""
import numpy as np
from numpy.typing import NDArray
from src.mcda_methods.McdaMethod import McdaMethod
from src.mcda_methods.SawSum import SawSum
from src.mcda_methods.SawMinMax import SawMinMax
from src.mcda_methods.Promethee import Promethee
from numba import jit, float64, int64, boolean, types


class LikelihoodQISMAA:
    def __init__(self, n_alts: int, n_crits: int, probs_of_alts: NDArray[float], mcda_method: McdaMethod, alpha: float,
                 beta: float, p: float, q: float):
        """
        Parameters
        ----------
        n_alts : int
            the number of alternatives
        n_crits : int
            the number of criteria
        probs_of_alts : 1D-numpy-array
            observed probabilities of the alternatives
        mcda_method : McdaMethod
            given MCDA method
        alpha : float
            parameter alpha in QISMAA
        beta : float
            parameter beta in QISMAA
        p : float
            p of the lp-norm in the discrepancy function
        q : float
            power of the lp-norm in the discrepancy function
        """
        self.n_alts = n_alts
        self.n_crits = n_crits
        self.probs_of_alts = probs_of_alts
        self.mcda_method = mcda_method
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.q = q

        # ensure that the sum of probs is one
        self.probs_of_alts = probs_of_alts / np.sum(probs_of_alts)

        if self.p < 1:
            raise NameError('p must be at least 1.')

        if self.q < 1:
            raise NameError('q must be at least 1.')

    def log_density(self, theta: NDArray[float]) -> float:
        """
        This function splits the parameter vector into the weight vector and the performance matrix perfmat to compute
        the logarithm of the likelihood function.

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
        perfmat = theta[self.n_crits:].reshape(self.n_alts, self.n_crits)
        return self._log_density(w, perfmat)[0]

    def _log_density(self, w: NDArray[float], perfmat: NDArray[float]) -> [float, NDArray[float]]:
        """
        This function computes the likelihood function for a given weight vector w and a given performance matrix
        perfmat. The likelihood is the probability of the shares self.probs_of_alts given w and perfmat when applying
        the MCDA method self.mcda_method.
        We outsourced the actual computation in the functions _log_density_saw and _log_density_p2, as we can
        accelerate this function using numba. This is more cumbersome, but easier than jitting _log_density directly.

        Parameters
        ----------
        w : 1D-numpy-array
            the weight vector
        perfmat : 2D-numpy-array
            the raw performance matrix

        Returns
        -------
        float
            the logarithm of the likelihood function at point (w, perfmat)
        """
        if self.probs_of_alts.size == 0:
            return 0

        if isinstance(self.mcda_method, SawSum) or isinstance(self.mcda_method, SawMinMax):
            pmat = self.mcda_method.p_from_perfmat(perfmat)
            return _log_density_saw(w, pmat, self.probs_of_alts, self.n_crits, self.n_alts, self.alpha,
                                    self.beta, self.p, self.q)
        elif isinstance(self.mcda_method, Promethee):
            pmat = self.mcda_method.p_from_perfmat(perfmat)
            return _log_density_p2(w, pmat, self.probs_of_alts, self.n_crits, self.n_alts, self.alpha,
                                   self.beta, self.p, self.q)

        # For non-default MCDA methods
        # calculate performances
        p = self.mcda_method.p_from_perfmat(perfmat)
        values_of_alts = p @ w

        # get overall range of possible performances
        max_value = p.max()
        min_value = p.min()

        corr_factor = 1.0

        values_scaled = (values_of_alts - max_value)/(max_value - min_value)*corr_factor

        # model the likelihood by using softmax on the performance differences and computing a distance measure
        differences = (np.log(self.n_alts-1) - np.log(1/self.beta - 1)) * values_scaled

        # differences = self.n_alts * (performances - np.max(performances))
        # convert to probabilities or shares using a softmax approach
        synth_shares = np.exp(differences) / np.sum(np.exp(differences))

        #
        return -self.alpha*np.power(np.linalg.norm(synth_shares - self.probs_of_alts, ord=self.p), self.q), synth_shares


@jit(types.Tuple((float64, float64[:]))(float64[:], float64[:, :], float64[:], int64, int64, float64,
                                        float64, float64, float64), nopython=True)
def _log_density_saw(w_in, pmat_in, probs_of_alts, n_crits, n_alts, alpha: float, beta: float, p: float, q: float):
    # Numba is obviously rather sophisticated. Any "optimisations" as manual recoding of numpy-functions, in particular
    # np.linalg.norm, did not lead to any measurable speed-up compared to that version of the code.
    w = np.copy(w_in)
    pmat = np.copy(pmat_in)

    values_of_alts = pmat @ w

    # get overall range of possible performances
    max_value = pmat.max()
    min_value = pmat.min()

    p_aux = np.zeros(pmat.shape)
    aux_vec = np.zeros(n_crits)

    means_per_crit = np.sum(pmat, axis=0) / n_alts
    norm_per_crit = np.sum(np.abs(pmat), axis=0)
    for icol in range(n_crits):
        p_aux[:, icol] = (np.abs(pmat[:, icol] - means_per_crit[icol])) / norm_per_crit[icol]
        aux_vec[icol] = np.max(p_aux[:, icol])

    corr_factor = np.mean(aux_vec) * n_alts / (n_alts - 1)

    values_scaled = (values_of_alts - max_value)/(max_value - min_value)*corr_factor

    # model the likelihood by using softmax on the performance differences and computing a distance measure
    differences = (np.log(n_alts-1) - np.log(1/beta - 1)) * values_scaled

    # convert to probabilities or shares using a softmax approach
    synth_shares = np.exp(differences) / np.sum(np.exp(differences))

    return -alpha*np.power(np.linalg.norm(synth_shares - probs_of_alts, ord=p), q), synth_shares


@jit(types.Tuple((float64, float64[:]))(float64[:], float64[:, :], float64[:], int64, int64, float64, float64, float64, float64), nopython=True)
def _log_density_p2(w_in, pmat_in, probs_of_alts, n_crits, n_alts, alpha: float, beta: float, p: float, q: float):

    w = np.copy(w_in)
    pmat = np.copy(pmat_in)

    values_of_alts = pmat @ w

    # get overall range of possible performances
    max_value = pmat.max()
    min_value = pmat.min()

    values_scaled = (values_of_alts - max_value)/(max_value - min_value)*np.max(np.abs(pmat))

    # model the likelihood by using softmax on the performance differences and computing a distance measure
    differences = (np.log(n_alts-1) - np.log(1/beta - 1)) * values_scaled

    # convert to probabilities or shares using a softmax approach
    synth_shares = np.exp(differences) / np.sum(np.exp(differences))

    return -alpha*np.power(np.linalg.norm(synth_shares - probs_of_alts, ord=p), q), synth_shares

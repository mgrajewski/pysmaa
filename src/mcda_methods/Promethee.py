"""
@author: Matthias Grajewski, FH Aachen University of Applied Sciences
Luis Hasenauer, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""

import numpy as np
from numpy.typing import NDArray
from src.mcda_methods.McdaMethod import McdaMethod
from numba import jit, float64, int64, boolean


class Promethee(McdaMethod):
    def __init__(self, type_criterion: NDArray[int], params: NDArray[float], benefit: NDArray[bool]):
        """
        type_criterion: 1D-numpy-integer-array
            types of preference function criterion following Brans et al., How to select and how to rank projects: The
            PROMETHEE method, European Journal of Operational Research 24 (1986) 228-238,
                #. :math:`\\operatorname{P(x)} = 1, x > 0`; 0 elsewhere (piecewise constant)
                #. :math:`\\operatorname{P(x)} = 1, x > p`; 0 elsewhere (u-shape preference function)
                #. :math:`\\operatorname{P(x)} = px, x > 0`; 0 elsewhere (v-shape preference function)
                #. not yet implemented
                #. :math:`\\operatorname{P(x)} = max(0, min(\\frac{x-q}{(p-q)}, 1))` (linear between q and p, truncated
                to 0 and 1 elsewhere)
                #. :math:`\\operatorname{P(x)} = max(0, 1-exp(\\frac{1}{-x^2/(2s^2)}))` (sigmoid function)

        params : 2D-numpy-array
            contains the parameters for the preference functions for each criterion. Shape: (2, ncrit)

            type_criterion: 1   2   3   4   5   6
            params[0, :]    -   p   p   p   q   s
            params[1, :]    -   -   -   q   p   -

        benefit: 1D-numpy array (bool)
            if true, the i-th criterion is associated with a benefit, if false, with costs.
        """
        self.type_criterion = type_criterion
        self.params = params
        self.benefit = benefit

    def p_from_perfmat(self, perfmat: NDArray[float]) -> NDArray[float]:
        """
        This function computes from the raw performance matrix perfmat the Promethee-II-matrix :math:`P`. For details,
        we refer to the external documentation.
        We assume that perfmat is valid, i.e. does not contain duplicate rows. You can check this using the separate
        function check_p.
        We outsourced the actual computation in the function p_from_perfmat_aux, as we can accelerate this function
        using numba. Jitting p_from_perfmat directly is currently impossible with numba as we use abstract classes.

        Parameters
        ----------
        perfmat : 2D-numpy-array
            raw performance matrix

        Returns
        -------
        p: 2D-numpy array
            Promethee-II-matrix
        """

        return p_from_perfmat_aux(perfmat, self.benefit, self.type_criterion, self.params)

    def get_values_of_alts(self, perfmat: NDArray[float], weights: NDArray[float]) -> NDArray[float]:
        p = p_from_perfmat_aux(perfmat, self.benefit, self.type_criterion, self.params)
        return p @ weights


@jit(float64[:, :](float64[:, :], boolean[:], int64[:], float64[:, :]), nopython=True)
def p_from_perfmat_aux(perfmat, benefit, type_criterion, params):
    # number of alternatives and number of criteria
    (nalts, ncrit) = np.shape(perfmat)

    phat = np.empty((nalts, nalts, ncrit))

    # it is favourable to preallocate that vector
    aux_nalts = np.ones(nalts)

    # compute d_ikj = P(i,j) - P(k,j) and store it in phat
    for icrit in range(ncrit):
        aux2 = np.outer(perfmat[:, icrit], aux_nalts)
        phat[:, :, icrit] = aux2 - aux2.T

    # compute f_j(d_ikj)
    for icrit in range(ncrit):
        # usual criterion
        if type_criterion[icrit] == 1:

            aux = phat[:, :, icrit]

            for i in range(nalts):
                aux_vec = aux[:, i]
                aux_vec[aux_vec > 0] = 1
                aux_vec[aux_vec < 0.1] = 0
                aux[:, i] = aux_vec

            # the variant below is more elegant, of course, but Numba only
            # supports advanced indexing for 1d-arrays
            # aux[aux > 0] = 1
            # aux[aux < 0.1] = 0

            # the current criterion is associated with a benefit and needs to be maximized
            if benefit[icrit]:
                phat[:, :, icrit] = aux - aux.T
            # the current criterion is associated with a cost and needs to be minimized
            else:
                phat[:, :, icrit] = aux.T - aux

        elif type_criterion[icrit] == 2:
            aux = phat[:, :, icrit]

            p = params[0, icrit]
            for i in range(nalts):
                aux_vec = aux[:, i]
                # we implicitly assume p >= 0 here
                aux_vec[aux_vec <= p] = 0
                aux_vec[aux_vec > p] = 1.0
                aux[:, i] = aux_vec

            # the current criterion is associated with a benefit and needs to be maximized
            if benefit[icrit]:
                phat[:, :, icrit] = aux - aux.T
            # the current criterion is associated with a cost and needs to be minimized
            else:
                phat[:, :, icrit] = aux.T - aux

        elif type_criterion[icrit] == 3:
            p = params[0, icrit]
            phat[:, :, icrit] = np.maximum(0, np.minimum(1.0, 1.0 / p * phat[:, :, icrit]))
            phat[:, :, icrit] = phat[:, :, icrit] - phat[:, :, icrit].T

            # if the criterion icrit is associated with a cost and not with a benefit, we minimize it aka
            # we maximize the negative cost
            if not benefit[icrit]:
                phat[:, :, icrit] = -phat[:, :, icrit]

        elif type_criterion[icrit] == 4:
            p = params[0, icrit]
            q = params[1, icrit]

            aux = phat[:, :, icrit]

            for i in range(nalts):
                for j in range(nalts):
                    if aux[i, j] < p:
                        aux[i, j] = 0.0
                    elif aux[i, j] > q:
                        aux[i, j] = 1.0
                    else:
                        aux[i, j] = 0.5

            # the current criterion is associated with a benefit and needs to be maximized
            if benefit[icrit]:
                phat[:, :, icrit] = aux - aux.T
            # the current criterion is associated with a cost and needs to be minimized
            else:
                phat[:, :, icrit] = aux.T - aux

        elif type_criterion[icrit] == 5:
            # The first row is the lower bound, the second row is the upper bound.
            q = params[0, icrit]
            p = params[1, icrit]

            phat[:, :, icrit] = np.maximum(0.0, np.minimum(1.0, 1 / (p - q) * (phat[:, :, icrit] - q)))
            phat[:, :, icrit] = phat[:, :, icrit] - phat[:, :, icrit].T

            if not benefit[icrit]:
                phat[:, :, icrit] = -phat[:, :, icrit]

        # Gaussian criterion (sigmoid function)
        elif type_criterion[icrit] == 6:
            s = params[0, icrit]
            aux = np.maximum(phat[:, :, icrit], 0)
            aux2 = 1 - np.exp(-aux * aux / (2.0 * s ** 2))

            if benefit[icrit]:
                phat[:, :, icrit] = aux2 - aux2.T
            else:
                phat[:, :, icrit] = aux2.T - aux2

        else:
            raise NameError('Invalid choice of criterion inside Promethee II')

    # sum over alternatives
    p = np.sum(phat, 1)

    # scale P accordingly
    p = 1 / (nalts - 1) * p

    return p

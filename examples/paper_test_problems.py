"""
@author: Matthias Grajewski, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""

import sys
from os import path

if __name__ == '__main__':

    path_to_pysmaa = "..\\"

    # if a path to pysmaa is provided, read it and append if, if not
    # already present in sys.path
    if path_to_pysmaa != '':
        if not any(path.normcase(sp) == path_to_pysmaa for sp in sys.path):
            sys.path.append(f'{path_to_pysmaa}/src')

import numpy as np

from src.priors.weights.NormalWeightsPrior import NormalWeightsPrior


def tests(i_test):
    """
    This function defines the Test Problems from "Reverse-engineering stakeholders' attitudes from observed preferences
    and quantitative data by Inverse Stochastic Multicriteria Acceptability Analysis".
    """

    weights_prior = None
    perfmat_mean = None
    perfmat_types = None
    perfmat_params = None
    all_probs_of_alts = None

    if i_test == 0:
        # Test Problem 3.1 from "Reverse-engineering stakeholders' attitudes from observed preferences and quantitative
        # data by Inverse Stochastic Multicriteria Acceptability Analysis"
        perfmat_mean = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float64)

        # uniformly distributed performance matrix, 0.1 absolute uncertainty
        perfmat_types = np.ones(perfmat_mean.shape, dtype=np.int64)
        perfmat_params = -0.1*np.ones(perfmat_mean.shape)

        # prior for weights
        weights_prior = NormalWeightsPrior(np.array([0.4, 0.6]), 0.01*np.eye(2))

        # observed probabilities
        all_probs_of_alts = [np.array([1.0, 0.0, 0.0]), np.array([0.9, 0.1, 0.0]), np.array([0.75, 0.25, 0.0]),
                             np.array([0.6, 0.4, 0.0]), np.array([0.35, 0.35, 0.3])]

    if i_test == 1:
        # Test Problem 3.2 from "Reverse-engineering stakeholders' attitudes from observed preferences and quantitative
        # data by Inverse Stochastic Multicriteria Acceptability Analysis"
        perfmat_mean = np.array([[2./3., 0], [0, 2./3.], [1./3, 1./3]], dtype=np.float64)

        # uniformly distributed performance matrix, 0.2 absolute uncertainty
        perfmat_types = np.ones(perfmat_mean.shape, dtype=np.int64)
        perfmat_params = -0.2 * np.ones(perfmat_mean.shape)

        # prior for weights
        weights_prior = NormalWeightsPrior(np.array([0.4, 0.6]), 0.01 * np.eye(2))

        # observed probabilities
        all_probs_of_alts = [np.array([1.0, 0.0, 0.0]), np.array([0.75, 0.25, 0.0]),
                             np.array([0.5, 0.5, 0.0]), np.array([0.4, 0.4, 0.2])]

    elif i_test == 3:
        # Test Problem 3.3 from "Reverse-engineering stakeholders' attitudes from observed preferences and quantitative
        # data by Inverse Stochastic Multicriteria Acceptability Analysis"
        perfmat_mean = np.array([[0.75, 0, 0], [0, 0.75, 0], [0, 0, 0.75], [0.25, 0.25, 0.25]], dtype=np.float64)

        # uniformly distributed performance matrix, 0.2 absolute uncertainty
        perfmat_types = np.ones(perfmat_mean.shape, dtype=np.int64)
        perfmat_params = -0.2*np.ones(perfmat_mean.shape)

        # prior for weights
        weights_prior = NormalWeightsPrior(np.array([0.25, 0.5, 0.25]), 0.03*np.eye(3))

        # observed probabilities
        all_probs_of_alts = [np.array([0.7, 0.1, 0.1, 0.1]), np.array([0.6, 0.15, 0.15, 0.1]),
                             np.array([0.5, 0.2, 0.2, 0.1]), np.array([0.3, 0.3, 0.3, 0.1])]


    elif i_test == 4:
        # Test Problem 3.4 from "Reverse-engineering stakeholders' attitudes from observed preferences and quantitative
        # data by Inverse Stochastic Multicriteria Acceptability Analysis"
        perfmat_mean = np.array([[0.7, 0.3], [0.3, 0.7], [0, 0], [0.3, 0.2], [0.2, 0.3]], dtype=np.float64)

        # performance matrix, no uncertainty here
        perfmat_types = np.zeros(perfmat_mean.shape, dtype=np.int64)
        perfmat_params = np.zeros(perfmat_mean.shape)

        # prior for weights
        weights_prior = NormalWeightsPrior(np.array([0.4, 0.6]), 0.01 * np.eye(2))

        # observed probabilities
        all_probs_of_alts = [np.array([0.35, 0.35, 0.0, 0.15, 0.15])]

    return perfmat_mean, perfmat_types, perfmat_params, all_probs_of_alts, weights_prior


if __name__ == '__main__':
    print('This is paper_test_cases')

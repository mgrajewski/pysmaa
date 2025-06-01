"""
This module provides an example of forward SMAA.

We aim at estimating the probability that alternative i takes the j-th place in the ranking of an actor.

@authors: Matthias Grajewski, FH Aachen University of Applied Sciences
          Luis Hasenauer, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""
import sys
from os import path
import numpy as np


# ---------------------- change this to run own analysis ----------------------
# json-File with all necessary data and parameters for computation
COMP_PARS_FILE = 'forward_smaa'
# ----------------------------------------------------------------------------

# If this module is executed stand-alone, we want to import pysmaa without hard-coding its location here. Instead, we
# provide the path to pysmaa\src in the json-control-file and read it here.
if __name__ == '__main__':
    import json as js

    with open(f'{COMP_PARS_FILE}.json') as file:
        parameter_dict = js.load(file)

    path_to_pysmaa = parameter_dict['path_to_pysmaa']

    # if a path to pysmaa is provided, read it and append if, if not
    # already present in sys.path
    if path_to_pysmaa != '':
        if not any(path.normcase(sp) == path_to_pysmaa for sp in sys.path):
            sys.path.append(f'{path_to_pysmaa}/src')

# now, we can import all the functions from pysmaa
from src.mcda_methods.Promethee import Promethee

from src.priors.weights.NormalWeightsPrior import NormalWeightsPrior
from src.priors.perfmat.UncertainPerfMatPrior import UncertainPerfMatPrior

from read_from_Excel import read_perfmat_from_excel, read_rankings_from_excel, read_weights_from_excel
from inoutput import read_comp_pars


def forward_smaa(comp_pars_file):
    """
    Parameters
    ----------
    comp_pars_file : json-file
        It contains all necessary information for running the analyses. For details, we refer to the documentation
        provided in inoutput.py

    Returns
    -------
    p_alts : matrix containing the probabilities of the alternatives for a certain position in the ranking
    mean_scores_of_alts : mean value of the values associated with the alternatives in SMAA
    var_scores_of_alts : variance of the values associated with the alternatives in SMAA
    """

    # ------ read all problem-related parameters --------

    # read all computational parameters from the json-file
    this_pars = read_comp_pars(comp_pars_file)

    # read information related to characteristics/performances
    perfmat, perfmat_types, perfmat_params, benefit, criteria_dummy, alternatives = \
        read_perfmat_from_excel(this_pars.input_file_P, 'Characteristics')

    # read the rankings, the names of the stakeholders and the mean values of the weights
    criteria, stakeholder, weights_mean = read_weights_from_excel(this_pars.input_file_P)

    nalts, ncrit = perfmat.shape

    # It is not possible currently to read distributional information on the weights from Excel. This is as in contrast
    # to the values of the performance matrix, which are usually independent, the weights are dependent due to
    # normalisation, which makes coding the corresponding distributions much more complicated.

    # hard-code covariance matrix
    cov_mat_weights = 0.05*np.eye(ncrit)


    # type of preference function in Promethee II
    type_criterion = this_pars.promethee_pref_func_type*np.ones(ncrit, dtype=np.int64)

    parameters = np.zeros((2, ncrit))
    parameters[0, :] = this_pars.promethee_c
    parameters[1, :] = this_pars.promethee_q

    # We are going to use Promethee 2 as MCDA method
    mcda_method = Promethee(type_criterion, parameters, benefit)

    # number of samples
    n_samples = this_pars.n_samples

    p_alts = np.zeros((nalts, nalts))

    # Define the distribution for the weights. We consider stakeholder_1 only.
    prior_weights = NormalWeightsPrior(weights_mean[0], cov_mat_weights)

    # sample weights according to that distribution
    samples_weights = prior_weights.sample(n_samples)

    # sample instances of perfmat for statistics
    prior_perfmat = UncertainPerfMatPrior(perfmat, perfmat_types, perfmat_params)
    samples_perfmat = prior_perfmat.sample(n_samples)

    scores_of_alts = np.zeros((n_samples, nalts))

    # compute the Promethee 2-matrices from these samples
    for isample in range(n_samples):
        pmat = mcda_method.p_from_perfmat(samples_perfmat[isample])

        perf = pmat@samples_weights[isample]

        ranking = np.argsort(-perf)
        scores_of_alts[isample, :] = perf

        p_alts[np.arange(0, nalts), ranking] = p_alts[np.arange(0, nalts), ranking] + 1

    p_alts = p_alts/n_samples

    # compute variance and mean
    mean_scores_of_alts = np.mean(scores_of_alts, axis=0)
    var_scores_of_alts = np.var(scores_of_alts, axis=0)

    return p_alts, mean_scores_of_alts, var_scores_of_alts


if __name__ == '__main__':
    p_alts, mean_scores_of_alts, var_scores_of_alts = forward_smaa(COMP_PARS_FILE)
    print(p_alts)

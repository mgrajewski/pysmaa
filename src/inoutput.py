# -*- coding: utf-8 -*-
"""
@author: Matthias Grajewski, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""

import json as js


class CompPars:
    """
    This class contains all relevant information on the SMAA algorithm as the underlying MCDA model along with its
    parameters, sampling, etc.
    It is our philosophy to separate problem-related and algorithm-related information. The former one is stored in
    Excel files and the latter one in a json-file.
    """
    def __init__(self):
        # general parameters

        # type of MCDA model
        self.mcda_type = 'Promethee'

        # in case of Promethee: type of preference function (should be a vector in future releases, as it may differ
        # from criterion to criterion)
        self.promethee_pref_func_type = 6

        # in case of Promethee: parameter c (should be a vector in future releases, as it may differ from criterion to
        # criterion)
        self.promethee_c = 1

        # in case of Promethee: parameter q (should be a vector in future releases, as it may differ from criterion to
        # criterion)
        self.promethee_q = 0

        # input and output files

        # Excel file containing the preference matrix
        self.input_file_P = ''

        # Excel file containing rankings
        self.input_file_ranks = ''

        # Excel file containing shares
        self.input_file_shares = ''

        # Excel file containing bounds for weights
        self.input_file_bounds = ''

        # Excel file for the output
        self.output_file = ''

        # absolute path to the main directory of pySMAA
        self.abs_path_to_pysmaa = ''

        # data for stochastic analysis
        self.n_samples = 100000

        # For P fixed and using SAW or Promethee as MCDA method, the set of weights leading to the given ranking is a
        # polytope. In this case, it may make sense to sample the weights according to a Gaussian distribution truncated
        # to that polytope. The corresponding covariance matrix comes from inscribing the volume-maximal ellipsoid in
        # the polytope. However, it may be necessary to scale that covariance matrix. THe corresponding scaling factor
        # is stored here.
        self.scale = 0.5

    def __str__(self):
        variables = vars(self)
        string_of_vars = ''
        for p in variables:
            string_of_vars += f'{p}: {variables[p]}\n'
        return string_of_vars


def read_comp_pars(comp_pars_file: str) -> CompPars:
    """
    This function reads parameters from a json-file and writes them into
    comp_pars object.

    Parameters
    ----------
    comp_pars_file (str): Name of json file to read from

    Returns
    -------
    comp_pars_object (comp_pars): Object with parameters from json file
    """
    with open(f'{comp_pars_file}.json') as file:
        parameter_dict = js.load(file)
    comp_pars_object = CompPars()
    for param in parameter_dict:
        object.__setattr__(comp_pars_object, str(param), parameter_dict[param])
    return comp_pars_object


def write_data(comp_pars_object: CompPars, comp_pars_file: str) -> None:
    """
    This function writes parameters from the comp_pars object to a json file.

    Parameters
    ----------
    comp_pars_object (comp_pars): Object to read parameters from
    comp_pars_file (str): Filename to write parameters to
    """
    with open(f'{comp_pars_file}.json', 'w') as file:
        js.dump(vars(comp_pars_object), file)


if __name__ == '__main__':
    test_write = CompPars()
    test_write.output_file = 'Hello World'
    write_data(test_write, 'test')
    test_read = read_comp_pars('test')
    # Output File should contain 'Hello World'
    print(test_read)

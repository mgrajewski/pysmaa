# -*- coding: utf-8 -*-
"""
@author: D. Plottka, L. Hasenauer, M. Grajewski, FH Aachen University of Applied Sciences
"""
import json as js


class comp_pars:
    """
    Documentation yet to come
    """
    def __init__(self):
        # general parameters
        self.mcda_type = 'Promethee'
        self.promethee_pref_func_type = 6
        self.promethee_c = 1
        self.promethee_q = 0

        # input and output files
        self.input_file_P = ''
        self.input_file_ranks = ''
        self.input_file_bounds = ''
        self.output_file = ''

        # absolute path to the main directory of pySMAA
        self.abs_path_to_pysmaa = ''

        # data for stochastic analysis
        self.n_samples = 100000
        self.scale = 0.5

    def __str__(self):
        variables= vars(self)
        string_of_vars = ''
        for p in variables:
            string_of_vars += f'{p}: {variables[p]}\n'
        return string_of_vars


def read_comp_pars(comp_pars_file: str) -> comp_pars:
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
    comp_pars_object = comp_pars()
    for param in parameter_dict:
        object.__setattr__(comp_pars_object, str(param), parameter_dict[param])
    return comp_pars_object


def write_data(comp_pars_object: comp_pars, comp_pars_file: str) -> None:
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
    test_write = comp_pars()
    test_write.output_file = 'Hello World'
    write_data(test_write, 'test')
    test_read = read_comp_pars('test')
    #Output File sollte 'Hello World' sein
    print(test_read)

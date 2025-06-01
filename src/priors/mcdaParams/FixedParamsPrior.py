"""
@author: Luis Hasenauer, FH Aachen University of Applied Sciences
Matthias Grajewski, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""

from numpy.typing import NDArray


class FixedParamsPrior:
    """
    This class represents a prior distribution of the MCDA parameters if these are fixed constants.
    """
    def __init__(self, params: NDArray[float]):
        """
        Parameters
        ----------
        params : 2D-numpy-array
            values of the parameters of the MCDA model
        """
        self.params = params

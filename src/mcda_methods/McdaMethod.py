"""
@author: Matthias Grajewski, FH Aachen University of Applied Sciences
Luis Hasenauer, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""
from abc import ABC, abstractmethod
from numpy.typing import NDArray


class McdaMethod(ABC):
    @abstractmethod
    def p_from_perfmat(self, perfmat: NDArray[float]) -> NDArray[float]:
        pass

    @abstractmethod
    def get_values_of_alts(self, perfmat: NDArray[float], weights: NDArray[float]) -> NDArray[float]:
        pass

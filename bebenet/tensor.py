from typing import Any, List
import numpy as np


class Tensor(np.ndarray):

    @staticmethod
    def from_array(array: List[Any]) -> 'Tensor':
        return np.array(array)

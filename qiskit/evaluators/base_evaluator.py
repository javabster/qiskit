# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Evaluator class base class
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np

from qiskit.providers import BackendV1 as Backend


class BaseEvaluator(ABC):
    def __init__(self, backend=Optional[Backend]):
        self._backend = backend

    @abstractmethod
    def evaluate(self, params: Optional[Union[List[float], np.ndarray]]):
        NotImplemented

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
Base Postprocessing class
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

from qiskit.evaluators.backends import ShotResult
from qiskit.evaluators.results.base_result import BaseResult
from qiskit.result import Result


class BasePostprocessing(ABC):
    """
    Base class for postprocessing.
    """

    def __call__(self, result, metadata):
        return self.execute(result, metadata)

    @abstractmethod
    def execute(self, result: Union[ShotResult, Result], metadata: list[dict]) -> BaseResult:
        """
        TODO
        """
        return NotImplemented

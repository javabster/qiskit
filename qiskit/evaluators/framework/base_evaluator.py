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
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

from qiskit.evaluators.backends import (
    BackendWrapper,
    BaseBackendWrapper,
    ShotBackendWrapper,
)
from qiskit.evaluators.results.base_result import BaseResult
from qiskit.providers import BackendV1 as Backend
from qiskit.providers import Options


class BaseEvaluator(ABC):
    """
    Base class for evaluator.
    """
    _default_run_options = Options()

    def __init__(self, backend: Union[Backend, BaseBackendWrapper, ShotBackendWrapper]):
        """
        Args:
            backend: backend
        """
        self._backend: Union[BaseBackendWrapper, ShotBackendWrapper]
        if isinstance(backend, ShotBackendWrapper):
            self._backend = backend
        else:
            self._backend = BackendWrapper.from_backend(backend)
        self._run_options = self._default_run_options

    @property
    def run_options(self) -> Options:
        """Return options values for the evaluator.
        Returns:
            run_options
        """
        return self._run_options

    def set_run_options(self, **fields) -> BaseEvaluator:
        """Set options values for the evaluator.

        Args:
            fields: The fields to update the options
        Returns:
            self
        """
        self._run_options.update_options(**fields)
        return self

    @abstractmethod
    def evaluate(
        self, parameters: Optional[Union[list[float], np.ndarray]], **run_options
    ) -> BaseResult:
        """
        TODO
        """
        return NotImplemented

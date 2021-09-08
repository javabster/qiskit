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

import copy
import sys
from typing import Any, Optional, Union, cast

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.evaluators.backends import (
    BackendWrapper,
    BaseBackendWrapper,
    ShotBackendWrapper,
    ShotResult,
)
from qiskit.evaluators.results import CompositeResult
from qiskit.evaluators.results.base_result import BaseResult
from qiskit.providers import BackendV1 as Backend
from qiskit.providers import Options
from qiskit.result import Result

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


class Postprocessing(Protocol):
    """Postprocessing Callback Protocol (PEP544)"""

    def __call__(self, result: Union[ShotResult, dict]) -> BaseResult:
        ...


class BaseEvaluator:
    """
    Base class for evaluator.
    """

    _default_run_options = Options()
    _default_transpile_options = Options()

    def __init__(
        self,
        backend: Union[Backend, BaseBackendWrapper],
        postprocessing: Postprocessing,
        transpile_options: Optional[dict] = None,
    ):
        """
        Args:
            backend: backend
        """
        self._backend: BaseBackendWrapper
        if isinstance(backend, ShotBackendWrapper):
            self._backend = backend
        else:
            self._backend = BackendWrapper.from_backend(backend)
        self._run_options = self._default_run_options

        self._transpile_options = self._default_transpile_options
        if transpile_options is not None:
            self.set_transpile_options(**transpile_options)

        self._transpiled_circuits: Optional[list[QuantumCircuit]] = None

        self._postprocessing = postprocessing

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

    @property
    def transpile_options(self) -> Options:
        """Return the transpiler options for transpiling the circuits."""
        return self._transpile_options

    def set_transpile_options(self, **fields) -> BaseEvaluator:
        """Set the transpiler options for transpiler.
        Args:
            fields: The fields to update the options
        Returns:
            self
        """
        self._transpile_options.update_options(**fields)
        return self

    @property
    def backend(self) -> Backend:
        """Backend

        Returns:
            backend
        """
        return self._backend.backend

    @property
    def transpiled_circuits(self) -> Optional[list[QuantumCircuit]]:
        """
        Transpiled quantum circuits produced by preprocessing

        Returns:
            List of the transpiled quantum circuit
        """
        return self._transpiled_circuits

    def evaluate(
        self,
        parameters: Optional[
            Union[
                list[float],
                list[list[float]],
                np.ndarray[Any, np.dtype[np.float64]],
            ]
        ] = None,
        had_transpiled=True,
        **run_options,
    ) -> BaseResult:
        """
        TODO
        """
        run_opts = copy.copy(self.run_options)
        run_opts.update_options(**run_options)
        run_opts_dict = run_opts.__dict__

        # TODO: support Aer parameter bind after https://github.com/Qiskit/qiskit-aer/pull/1317
        if parameters is None:
            circuits = self.transpiled_circuits
        else:
            parameters = np.asarray(parameters, dtype=np.float64)
            if parameters.ndim == 1:
                circuits = [
                    circ.bind_parameters(parameters)  # type: ignore
                    for circ in self.transpiled_circuits
                ]
            elif parameters.ndim == 2:
                circuits = [
                    circ.bind_parameters(parameter)
                    for parameter in parameters
                    for circ in self.transpiled_circuits
                ]
            else:
                raise TypeError("The number of array dimension must be 1 or 2.")

        if not had_transpiled:
            transpile_opts_dict = self.transpile_options.__dict__
            circuits = cast(
                list[QuantumCircuit], transpile(circuits, self.backend, **transpile_opts_dict)
            )

        results = self._backend.run_and_wait(circuits, **run_opts_dict)

        if parameters is None or isinstance(parameters, np.ndarray) and parameters.ndim == 1:
            if isinstance(results, Result):
                return self._postprocessing(results.data(0))
            return self._postprocessing(results)
        if isinstance(results, Result):
            postprocessed = [self._postprocessing(results.data(i)) for i in range(len(parameters))]
        else:
            postprocessed = [
                self._postprocessing(
                    results[
                        i * len(self.transpiled_circuits) : (i + 1) * len(self.transpiled_circuits)
                    ]
                )
                for i in range(len(parameters))
            ]
        return CompositeResult(postprocessed)

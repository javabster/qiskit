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
from typing import Optional, Union, cast

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

    def __call__(self, result: Union[ShotResult, Result]) -> BaseResult:
        ...


class BaseEvaluator:
    """
    Base class for evaluator.
    """

    _default_run_options = Options()
    _default_transpile_options = Options()

    def __init__(
        self,
        backend: Union[Backend, BaseBackendWrapper, ShotBackendWrapper],
        postprocessing: Postprocessing,
        transpile_options: Optional[dict] = None,
    ):
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

        self._transpile_options = self._default_transpile_options
        if transpile_options is not None:
            self.set_transpile_options(**transpile_options)

        self._transpiled_circuits: Optional[list[QuantumCircuit]] = None
        self._metadata: Optional[list[dict]] = None
        self._num_circuits: Optional[list[int]] = None

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
        parameters: Optional[Union[list[float], list[list[float]]]] = None,
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
        elif isinstance(parameters, list) and isinstance(parameters[0], list):
            parameters = cast(list[list[float]], parameters)
            if self._num_circuits is None:
                circuits = [
                    circ.bind_parameters(params)
                    for params in parameters
                    for circ in self.transpiled_circuits
                ]
                self._num_circuits = [len(self.transpiled_circuits)] * len(parameters)
                self._metadata = sum([self._metadata] * len(parameters), [])
            else:
                if len(parameters) != len(self._num_circuits):
                    raise TypeError("Length is different.")

                flatten_parameters: list[list[float]] = sum(
                    [[params] * num for params, num in zip(parameters, self._num_circuits)], []
                )
                circuits = [
                    circ.bind_parameters(param)
                    for param, circ in zip(flatten_parameters, self.transpiled_circuits)
                ]
        elif isinstance(parameters, list) and isinstance(parameters[0], (float, int)):
            parameters = cast(list[float], parameters)
            circuits = [circ.bind_parameters(parameters) for circ in self.transpiled_circuits]
        else:
            raise TypeError()

        if not had_transpiled:
            transpile_opts_dict = self.transpile_options.__dict__
            circuits = cast(
                list[QuantumCircuit], transpile(circuits, self.backend, **transpile_opts_dict)
            )

        results = self._backend.run_and_wait(circuits, **run_opts_dict)

        if (
            isinstance(parameters, list) and isinstance(parameters[0], list)
        ) or self._num_circuits is not None:
            postprocessed_results = []
            accum = 0
            for num_circuit in self._num_circuits:
                postprocessed_results.append(
                    self._postprocessing(
                        results[accum : accum + num_circuit],
                    )
                )
                accum += num_circuit

            return CompositeResult(postprocessed_results)
        return self._postprocessing(results)

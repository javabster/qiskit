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

    def __init__(
        self,
        backend: Union[Backend, BaseBackendWrapper],
        postprocessing: Postprocessing,
        transpile_options: Optional[dict] = None,
        append: bool = False,
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
        self._run_options = Options()

        self._transpile_options = Options()
        if transpile_options is not None:
            self.set_transpile_options(**transpile_options)

        self._preprocessed_circuits: Optional[list[QuantumCircuit]] = None
        self._transpiled_circuits: Optional[list[QuantumCircuit]] = None

        self._postprocessing = postprocessing

        self.append = append
        self.history: list[BaseResult] = []

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
        self._preprosessed_circuits = None
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
    def preprocessed_circuits(self) -> Optional[list[QuantumCircuit]]:
        """
        Preprocessed quantum circuits produced by preprocessing

        Returns:
            List of the transpiled quantum circuit
        """
        return self._preprocessed_circuits

    @property
    def transpiled_circuits(self) -> list[QuantumCircuit]:
        """
        Transpiled quantum circuits.

        Returns:
            List of the transpiled quantum circuit
        """
        if self._transpiled_circuits is None:
            self._transpile()
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
        **run_options,
    ) -> BaseResult:
        """
        TODO
        """
        # Bind parameters
        # TODO: support Aer parameter bind after https://github.com/Qiskit/qiskit-aer/pull/1317
        if parameters is None:
            bound_circuits = self.transpiled_circuits
        else:
            parameters = np.asarray(parameters, dtype=np.float64)
            if parameters.ndim == 1:
                bound_circuits = [
                    circ.bind_parameters(parameters)  # type: ignore
                    for circ in self.transpiled_circuits
                ]
            elif parameters.ndim == 2:
                bound_circuits = [
                    circ.bind_parameters(parameter)
                    for parameter in parameters
                    for circ in self.transpiled_circuits
                ]
            else:
                raise TypeError("The number of array dimension must be 1 or 2.")

        # Run
        run_opts = copy.copy(self.run_options)
        run_opts.update_options(**run_options)

        results = self._backend.run_and_wait(bound_circuits, **run_opts.__dict__)

        if parameters is None or isinstance(parameters, np.ndarray) and parameters.ndim == 1:
            if isinstance(results, Result):
                ret_result = self._postprocessing(results.data(0))
            else:
                ret_result = self._postprocessing(results)
        else:

            if isinstance(results, Result):
                postprocessed = [
                    self._postprocessing(results.data(i)) for i in range(len(parameters))
                ]
            else:
                postprocessed = [
                    self._postprocessing(
                        results[
                            i
                            * len(self.transpiled_circuits) : (i + 1)
                            * len(self.transpiled_circuits)
                        ]
                    )
                    for i in range(len(parameters))
                ]
            ret_result = CompositeResult(postprocessed)

        if self.append:
            self.history.append(ret_result)

        return ret_result

    def _transpile(self):
        if isinstance(self.preprocessed_circuits, tuple):
            # 1. transpile a common circuit
            transpiled_state = self.preprocessed_circuits[0].copy()
            num_qubits = transpiled_state.num_qubits
            transpiled_state.measure_all()
            transpiled_state = cast(
                QuantumCircuit,
                transpile(transpiled_state, self.backend, **self.transpile_options.__dict__),
            )
            bit_map = {bit: index for index, bit in enumerate(transpiled_state.qubits)}
            layout = [bit_map[qr[0]] for _, qr, _ in transpiled_state[-num_qubits:]]
            transpiled_state.remove_final_measurements()
            # 2. transpile diff circuits
            diff_circuits = self.preprocessed_circuits[1]
            transpile_opts = copy.copy(self.transpile_options)
            transpile_opts.update_options(initial_layout=layout)
            diff_circuits = cast(
                list[QuantumCircuit],
                transpile(diff_circuits, self.backend, **transpile_opts.__dict__),
            )
            # 3. combine
            transpiled_circuits = []
            for diff_circuit in diff_circuits:
                transpiled_circuit = transpiled_state.copy()
                for creg in diff_circuit.cregs:
                    if creg not in transpiled_circuit.cregs:
                        transpiled_circuit.add_register(creg)
                for inst, qargs, cargs in diff_circuit.data:
                    transpiled_circuit.append(inst, qargs, cargs)
                transpiled_circuit.metadata = diff_circuit.metadata
                transpiled_circuits.append(transpiled_circuit)
            self._transpiled_circuits = transpiled_circuits
        else:
            self._transpiled_circuits = cast(
                list[QuantumCircuit],
                transpile(
                    self.preprocessed_circuits, self.backend, **self.transpile_options.__dict__
                ),
            )

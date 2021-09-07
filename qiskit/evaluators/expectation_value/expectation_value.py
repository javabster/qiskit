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
Expectation value class
"""
from __future__ import annotations

import sys
from typing import Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.evaluators.backends import (
    BaseBackendWrapper,
    ShotBackendWrapper,
    ShotResult,
)
from qiskit.evaluators.framework import BaseEvaluator, BasePreprocessing
from qiskit.evaluators.results import ExpectationValueResult
from qiskit.extensions import Initialize
from qiskit.opflow import PauliSumOp
from qiskit.providers import BackendV1 as Backend
from qiskit.providers import Options
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import Result

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


class Preprocessing(Protocol):
    """Preprocessing Callback Protocol (PEP544)"""

    def __call__(
        self, state: QuantumCircuit, observable: SparsePauliOp
    ) -> tuple[list[QuantumCircuit], list[dict]]:
        ...


class Postprocessing(Protocol):
    """Postprocessing Callback Protocol (PEP544)"""

    def __call__(self, result: Union[ShotResult, Result]) -> ExpectationValueResult:
        ...


class ExpectationValue(BaseEvaluator):
    """
    Expectation Value class
    """

    def __init__(
        self,
        preprocessing: Preprocessing,
        postprocessing: Postprocessing,
        state: Union[QuantumCircuit, Statevector],
        observable: Union[BaseOperator, PauliSumOp],
        backend: Union[Backend, BaseBackendWrapper, ShotBackendWrapper],
    ):
        """ """
        super().__init__(backend, postprocessing)
        self._preprocessing = preprocessing
        self._state = self._init_state(state)
        self._observable = self._init_observable(observable)

    @property
    def state(self) -> QuantumCircuit:
        """Quantum Circuit that represents quantum state.

        Returns:
            quantum state
        """
        return self._state

    @state.setter
    def state(self, state: Union[QuantumCircuit, Statevector]):
        self._transpiled_circuits = None
        self._metadata = None
        self._state = self._init_state(state)

    @property
    def observable(self) -> SparsePauliOp:
        """
        SparsePauliOp that represents observable

        Returns:
            observable
        """
        return self._observable

    @observable.setter
    def observable(self, observable: Union[BaseOperator, PauliSumOp]):
        self._transpiled_circuits = None
        self._metadata = None
        self._observable = self._init_observable(observable)

    @property
    def transpile_options(self) -> Options:
        """
        Options for transpile

        Returns:
            transpile options
        """
        if isinstance(self._preprocessing, BasePreprocessing):
            return self._preprocessing.transpile_options
        return super().transpile_options

    def set_transpile_options(self, **fields) -> ExpectationValue:
        """Set the transpiler options for transpiler.

        Args:
            fields: The fields to update the options
        Returns:
            self
        """
        self._transpiled_circuits = None
        self._metadata = None
        if isinstance(self._preprocessing, BasePreprocessing):
            self._preprocessing.set_transpile_options(**fields)
        else:
            super().set_transpile_options(**fields)
        return self

    @property
    def transpiled_circuits(self) -> list[QuantumCircuit]:
        """
        Transpiled quantum circuits produced by preprocessing

        Returns:
            List of the transpiled quantum circuit
        """
        if self._transpiled_circuits is None:
            self._transpiled_circuits, self._metadata = self._preprocessing(
                self.state, self.observable
            )
        return super().transpiled_circuits

    @staticmethod
    def _init_state(state: Union[QuantumCircuit, Statevector]) -> QuantumCircuit:
        if isinstance(state, QuantumCircuit):
            return state
        statevector = Statevector(state)
        qc = QuantumCircuit(statevector.num_qubits)
        qc.append(
            Initialize(state.data, statevector.num_qubits), list(range(statevector.num_qubits))
        )
        return qc

    @staticmethod
    def _init_observable(observable: Union[BaseOperator, PauliSumOp]) -> SparsePauliOp:
        if isinstance(observable, PauliSumOp):
            if isinstance(observable.coeff, ParameterExpression):
                raise TypeError(
                    f"observable must have numerical coefficient, not {type(observable.coeff)}"
                )
            return observable.coeff * observable.primitive
        if isinstance(observable, SparsePauliOp):
            return observable
        if isinstance(observable, BaseOperator):
            return SparsePauliOp.from_operator(observable)

        raise TypeError(f"Unrecognized observable {type(observable)}")

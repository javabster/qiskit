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
from typing import TYPE_CHECKING, Optional, Union, cast

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.evaluators.backends import (
    BaseBackendWrapper,
    ShotBackendWrapper,
    ShotResult,
)
from qiskit.evaluators.framework import BaseEvaluator
from qiskit.evaluators.results import (
    CompositeResult,
    ExpectationValueArrayResult,
    ExpectationValueResult,
)
from qiskit.extensions import Initialize
from qiskit.opflow import PauliSumOp
from qiskit.providers import BackendV1 as Backend
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

if TYPE_CHECKING:
    from typing import Any

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


class Preprocessing(Protocol):
    """Preprocessing Callback Protocol (PEP544)"""

    def __call__(self, state: QuantumCircuit, observable: SparsePauliOp) -> list[QuantumCircuit]:
        ...


class Postprocessing(Protocol):
    """Postprocessing Callback Protocol (PEP544)"""

    def __call__(self, result: Union[ShotResult, dict]) -> ExpectationValueResult:
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
        append: bool = False,
    ):
        """ """
        super().__init__(backend=backend, postprocessing=postprocessing, append=append)
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
        self._observable = self._init_observable(observable)

    def set_transpile_options(self, **fields) -> ExpectationValue:
        """Set the transpiler options for transpiler.

        Args:
            fields: The fields to update the options
        Returns:
            self
        """
        self._transpiled_circuits = None
        super().set_transpile_options(**fields)
        return self

    @property
    def preprocessed_circuits(self) -> list[QuantumCircuit]:
        """
        Transpiled quantum circuits produced by preprocessing

        Returns:
            List of the transpiled quantum circuit
        """
        if self._preprocessed_circuits is None:
            self._preprocessed_circuits = self._preprocessing(self.state, self.observable)
        return super().preprocessed_circuits

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

    def evaluate(
        self,
        parameters: Optional[
            Union[
                list[float],
                list[list[float]],
                "np.ndarray[Any, np.dtype[np.float64]]",
            ]
        ] = None,
        **run_options,
    ) -> Union[ExpectationValueResult, ExpectationValueArrayResult]:
        res = super().evaluate(parameters, **run_options)
        if isinstance(res, CompositeResult):
            # TODO CompositeResult should be Generic
            values = np.array([r.value for r in res.items])  # type: ignore
            variances = np.array([r.variance for r in res.items])  # type: ignore
            confidence_intervals = np.array([r.confidence_interval for r in res.items])  # type: ignore
            return ExpectationValueArrayResult(values, variances, confidence_intervals)
        return cast(ExpectationValueResult, res)

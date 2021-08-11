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
Expectation value base class
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.opflow import PauliSumOp
from qiskit.providers import BackendV1 as Backend
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import Result

from .base_evaluator import BaseEvaluator
from .expectation_value_result import ExpectationValueResult


class BaseExpectationValue(BaseEvaluator, ABC):
    """ """

    def __init__(
        self,
        state: Union[QuantumCircuit, Statevector],
        observable: Union[BaseOperator, PauliSumOp],
        backend: Backend,
    ):
        """ """
        super().__init__(backend)
        self._state = self._init_state(state)
        self._observable = self._init_observable(observable)
        self._meas_circuits, self._metadata = self._preprocessing()

    @property
    def state(self):
        """ """
        return self._state

    @property
    def observable(self) -> SparsePauliOp:
        """ """
        return self._observable

    @property
    def evaluated_circuits(self) -> List[QuantumCircuit]:
        return self._meas_circuits

    @staticmethod
    def _init_state(state) -> QuantumCircuit:
        if isinstance(state, QuantumCircuit):
            return state
        statevector = Statevector(state)
        qc = QuantumCircuit(statevector.num_qubits)
        qc.initialize(state.data, list(range(statevector.num_qubits)))
        return qc

    @staticmethod
    def _init_observable(observable) -> SparsePauliOp:
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

    @abstractmethod
    def _preprocessing(self) -> Tuple[List[QuantumCircuit], List[dict]]:
        """ """

    @abstractmethod
    def _postprocessing(self, result: Result):
        """ """

    def evaluate(
        self, parameters: Optional[Union[List[float], np.ndarray]] = None
    ) -> ExpectationValueResult:

        if parameters is not None:
            bound_circuits = [circ.bind_parameters(parameters) for circ in self._meas_circuits]
            result = self._backend.run(bound_circuits).result()
        else:
            result = self._backend.run(self._meas_circuits).result()

        return self._postprocessing(result)

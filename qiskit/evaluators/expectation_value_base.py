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

from .evaluator_base import EvaluatorBase
from .expectation_value_result import ExpectationValueResult


class ExpectationValueBase(EvaluatorBase, ABC):
    """ """

    def __init__(
        self,
        state: Union[QuantumCircuit, Statevector],
        observable: Union[BaseOperator, PauliSumOp],
        backend: Backend,
    ):
        """ """

        super().__init__(backend)

        # Set state
        if isinstance(state, QuantumCircuit):
            self._state = state
        else:
            state = Statevector(state)
            self._state = QuantumCircuit(state.num_qubits)
            self._state.initialize(state.data, list(range(state.num_qubits)))

        # Set observable
        if isinstance(observable, PauliSumOp):
            if isinstance(observable.coeff, ParameterExpression):
                raise TypeError(
                    f"observable must have numerical coefficient, not {type(observable.coeff)}"
                )
            self._observable = observable.coeff * observable.primitive
        elif isinstance(observable, SparsePauliOp):
            self._observable = observable
        elif isinstance(observable, BaseOperator):
            self._observable = SparsePauliOp.from_operator(observable)
        else:
            raise TypeError(f"Unrecognized observable {type(observable)}")

        # preprocessing
        self._meas_circuits, self._metadata = self._preprocessing()

    @property
    def state(self):
        """ """
        return self._state

    @property
    def observable(self) -> SparsePauliOp:
        """ """
        return self._observable

    @abstractmethod
    def _preprocessing(self) -> Tuple[List[QuantumCircuit], List[dict]]:
        """ """

    @abstractmethod
    def _postprocessing(self, result: Result):
        """ """

    def evaluate(
        self, parameters: Optional[Union[List[float], np.ndarray]]
    ) -> ExpectationValueResult:
        binded_circuits = [circ.bind_parameters(parameters) for circ in self._meas_circuits]
        result = self._backend.run(binded_circuits).result()
        return self._postprocessing(result)

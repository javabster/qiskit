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

import copy
from typing import Optional, Union

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.evaluators.backends import BaseBackendWrapper, ShotBackendWrapper
from qiskit.evaluators.framework import (
    BaseEvaluator,
    BasePostprocessing,
    BasePreprocessing,
)
from qiskit.evaluators.results import ExpectationValueResult
from qiskit.extensions import Initialize
from qiskit.opflow import PauliSumOp
from qiskit.providers import BackendV1 as Backend
from qiskit.providers import Options
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class ExpectationValue(BaseEvaluator):
    """
    Expectation Value class
    """

    def __init__(
        self,
        preprocessing: BasePreprocessing,
        postprocessing: BasePostprocessing,
        state: Union[QuantumCircuit, Statevector],
        observable: Union[BaseOperator, PauliSumOp],
        backend: Union[Backend, BaseBackendWrapper, ShotBackendWrapper],
    ):
        """ """
        super().__init__(backend)
        self._preprocessing = preprocessing
        self._postprocessing = postprocessing
        self._state = self._init_state(state)
        self._observable = self._init_observable(observable)
        self._transpiled_circuits = None
        self._metadata = None

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
        return self._preprocessing.transpile_options

    def set_transpile_options(self, **fields) -> ExpectationValue:
        """Set the transpiler options for transpiler.

        Args:
            fields: The fields to update the options
        Returns:
            self
        """
        self._transpiled_circuits = None
        self._metadata = None
        self._preprocessing.set_transpile_options(**fields)
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
        return self._transpiled_circuits

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
        parameters: Optional[Union[list[float], np.ndarray]] = None,
        **run_options,
    ) -> ExpectationValueResult:

        run_opts = copy.copy(self.run_options)
        run_opts.update_options(**run_options)
        run_opts_dict = run_opts.__dict__

        # TODO: support Aer parameter bind after https://github.com/Qiskit/qiskit-aer/pull/1317
        if parameters is not None:
            bound_circuits = [circ.bind_parameters(parameters) for circ in self.transpiled_circuits]
            result = self._backend.run_and_wait(bound_circuits, **run_opts_dict)
        else:
            result = self._backend.run_and_wait(self.transpiled_circuits, **run_opts_dict)

        return self._postprocessing(result, self._metadata)

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

from typing import Optional, Union, cast

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.opflow import PauliSumOp
from qiskit.providers import BackendV1 as Backend
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import Result

from .base_expectation_value import BaseExpectationValue
from .processings.base_postprocessing import BasePostprocessing
from .processings.expectation_preprocessing import ExpectationPreprocessing
from .results.expectation_value_result import ExpectationValueResult


class ExactExpectationValue(BaseExpectationValue):
    def __init__(
        self,
        state: Union[QuantumCircuit, Statevector],
        observable: Union[BaseOperator, PauliSumOp],
        backend: Backend,
        transpile_options: Optional[dict] = None,
    ):
        super().__init__(
            ExactPreprocessing(backend=backend, transpile_options=transpile_options),
            ExactPostprocessing(),
            state=state,
            observable=observable,
            backend=backend,
        )


class ExactPreprocessing(ExpectationPreprocessing):
    def execute(
        self, state: QuantumCircuit, observable: SparsePauliOp
    ) -> tuple[list[QuantumCircuit], list[dict]]:
        # circuit transpilation
        transpiled_circuit: QuantumCircuit = cast(
            QuantumCircuit, transpile(state, self._backend, **self._transpile_options.__dict__)
        )  # TODO: option
        # TODO: final layout

        # TODO: need to check whether Aer exists or not
        transpiled_circuit.save_expectation_value_variance(
            operator=observable, qubits=range(transpiled_circuit.num_qubits)
        )
        return [transpiled_circuit], [{}]


class ExactPostprocessing(BasePostprocessing):
    def execute(self, result: Result, metadata) -> ExpectationValueResult:
        expval, variance = result.data()["expectation_value_variance"]

        return ExpectationValueResult(
            np.array(expval),
            np.array(variance),
            None,
        )

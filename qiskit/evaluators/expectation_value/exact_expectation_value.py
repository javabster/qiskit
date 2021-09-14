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

# pylint: disable=no-name-in-module, import-error

from __future__ import annotations

from typing import Union

from qiskit import QuantumCircuit
from qiskit.evaluators.backends import ShotResult
from qiskit.evaluators.framework import BasePostprocessing, BasePreprocessing
from qiskit.evaluators.results import ExpectationValueResult
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.opflow import PauliSumOp
from qiskit.providers import BackendV1 as Backend
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.utils import has_aer

from .expectation_value import ExpectationValue

if has_aer():
    from qiskit.providers.aer.library import SaveExpectationValueVariance


class ExactExpectationValue(ExpectationValue):
    """
    Calculates the expectation value exactly (i.e. without sampling error).
    """

    def __init__(
        self,
        state: Union[QuantumCircuit, Statevector],
        observable: Union[BaseOperator, PauliSumOp],
        backend: Backend,
        append: bool = False,
    ):
        if not has_aer():
            raise MissingOptionalLibraryError(
                libname="qiskit-aer",
                name="Aer provider",
                pip_install="pip install qiskit-aer",
            )

        super().__init__(
            ExactPreprocessing(),
            ExactPostprocessing(),
            state=state,
            observable=observable,
            backend=backend,
            append=append,
        )


class ExactPreprocessing(BasePreprocessing):
    """
    Preprocessing for :class:`ExactExpectationValue`.
    """

    def execute(self, state: QuantumCircuit, observable: SparsePauliOp) -> list[QuantumCircuit]:
        state_copy = state.copy()
        inst = SaveExpectationValueVariance(operator=observable)
        state_copy.append(inst, qargs=range(state_copy.num_qubits))
        return [state_copy]


class ExactPostprocessing(BasePostprocessing):
    """
    Postprocessing for :class:`ExactExpectationValue`.
    """

    def execute(self, result: Union[dict, ShotResult]) -> ExpectationValueResult:

        # TODO: validate

        expval, variance = result["expectation_value_variance"]
        return ExpectationValueResult(expval, variance, None)

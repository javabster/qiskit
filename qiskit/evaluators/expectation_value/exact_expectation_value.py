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

from typing import Optional, Union, cast

from qiskit import QuantumCircuit, transpile
from qiskit.evaluators.backends import ShotResult
from qiskit.evaluators.framework import BasePostprocessing, BasePreprocessing
from qiskit.evaluators.results import ExpectationValueResult
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.opflow import PauliSumOp
from qiskit.providers import BackendV1 as Backend
from qiskit.providers import Options
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import Result
from qiskit.utils import has_aer

from .expectation_value import ExpectationValue

if has_aer():
    from qiskit.providers.aer.library import SaveExpectationValueVariance


class ExactExpectationValue(ExpectationValue):
    """
    Calculates exact expectation value exactly (without sampling error).
    """

    def __init__(
        self,
        state: Union[QuantumCircuit, Statevector],
        observable: Union[BaseOperator, PauliSumOp],
        backend: Backend,
        transpile_options: Optional[dict] = None,
    ):
        self._preprocessing: BasePreprocessing
        if not has_aer():
            raise MissingOptionalLibraryError(
                libname="qiskit-aer",
                name="Aer provider",
                pip_install="pip install qiskit-aer",
            )

        super().__init__(
            ExactPreprocessing(backend=backend, transpile_options=transpile_options),
            ExactPostprocessing(),
            state=state,
            observable=observable,
            backend=backend,
        )

    @property
    def transpile_options(self) -> Options:
        """
        Options for transpile

        Returns:
            transpile options
        Raises:
            QiskitError: if preprocessing is not BasePreprocessing
        """
        return self._preprocessing.transpile_options

    def set_transpile_options(self, **fields) -> ExpectationValue:
        """Set the transpiler options for transpiler.

        Args:
            fields: The fields to update the options
        Returns:
            self
        Raises:
            QiskitError: if preprocessing is not BasePreprocessing
        """
        self._transpiled_circuits = None
        self._metadata = None
        self._preprocessing.set_transpile_options(**fields)
        return self


class ExactPreprocessing(BasePreprocessing):
    """
    Preprocessing for :class:`ExactExpectationValue`.
    """

    def execute(
        self, state: QuantumCircuit, observable: SparsePauliOp
    ) -> tuple[list[QuantumCircuit], list[dict]]:
        # circuit transpilation
        transpiled_circuit: QuantumCircuit = cast(
            QuantumCircuit, transpile(state, self._backend, **self._transpile_options.__dict__)
        )  # TODO: option
        # TODO: final layout

        # TODO: need to check whether Aer exists or not
        inst = SaveExpectationValueVariance(operator=observable)

        transpiled_circuit.append(inst, qargs=range(transpiled_circuit.num_qubits))
        return [transpiled_circuit], [{}]


class ExactPostprocessing(BasePostprocessing):
    """
    Postprocessing for :class:`ExactExpectationValue`.
    """

    def execute(self, result: Union[Result, ShotResult], metadata) -> ExpectationValueResult:

        if not isinstance(result, Result):
            raise TypeError(f"{self.__class__.__name__} does not support list[Counts] as an input.")

        expval, variance = result.data()["expectation_value_variance"]

        return ExpectationValueResult(
            expval,
            variance,
            None,
        )

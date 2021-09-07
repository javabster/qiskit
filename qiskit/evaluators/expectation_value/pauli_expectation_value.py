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

import logging
from typing import Optional, Union, cast

import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.evaluators.backends import (
    BackendWrapper,
    ShotBackendWrapper,
    ShotResult,
)
from qiskit.evaluators.framework import BasePostprocessing, BasePreprocessing
from qiskit.evaluators.results import ExpectationValueResult
from qiskit.opflow import PauliSumOp
from qiskit.providers import BackendV1 as Backend
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import Counts, Result

from .expectation_value import ExpectationValue

logger = logging.getLogger(__name__)


class PauliExpectationValue(ExpectationValue):
    """
    Evaluates expectaion value using pauli rotation gates.
    """

    def __init__(
        self,
        state: Union[QuantumCircuit, Statevector],
        observable: Union[BaseOperator, PauliSumOp],
        backend: Union[Backend, ShotBackendWrapper],
        transpile_options: Optional[dict] = None,
    ):
        super().__init__(
            PauliPreprocessing(
                BackendWrapper.to_backend(backend),
                transpile_options,
            ),
            PauliPostprocessing(),
            state=state,
            observable=observable,
            backend=ShotBackendWrapper.from_backend(backend),
        )


class PauliPreprocessing(BasePreprocessing):
    """
    Preprocessing for evaluation of expectation value using pauli rotation gates.
    """

    def execute(
        self, state: QuantumCircuit, observable: SparsePauliOp
    ) -> tuple[list[QuantumCircuit], list[dict]]:
        """
        TODO
        """

        # circuit transpilation
        transpiled_circuit: QuantumCircuit = cast(
            QuantumCircuit, transpile(state, self._backend, **self.transpile_options.__dict__)
        )
        # TODO: final layout

        circuits: list[QuantumCircuit] = []
        metadata: list[dict] = []
        for pauli, coeff in observable.label_iter():
            coeff = coeff.real.item() if np.isreal(coeff) else coeff.item()
            circuit = transpiled_circuit.copy()
            creg = ClassicalRegister(len(pauli))
            circuit.add_register(creg)
            for i, val in enumerate(reversed(pauli)):
                if val == "Y":
                    circuit.sdg(i)
                if val in ["Y", "X"]:
                    circuit.h(i)
                circuit.measure(i, i)
            circuit.metadata = {"basis": pauli, "coeff": coeff}
            circuits.append(circuit)
            metadata.append({"basis": pauli, "coeff": coeff})

        return circuits, metadata


class PauliPostprocessing(BasePostprocessing):
    """
    Postprocessing for evaluation of expectation value using pauli rotation gates.
    """

    def execute(
        self, result: Union[ShotResult, Result], metadata: list[dict]
    ) -> ExpectationValueResult:
        """
        TODO
        """
        if not isinstance(result, ShotResult):
            raise TypeError(f"result must be ShotResult, not {type(result)}.")

        data = result.counts

        combined_expval = 0.0
        combined_variance = 0.0
        combined_stderr = 0.0

        for dat, meta in zip(data, metadata):
            basis = meta.get("basis", None)
            diagonal = _pauli_diagonal(basis) if basis is not None else None
            coeff = meta.get("coeff", 1)
            # qubits = meta.get("qubits", None)
            shots = sum(dat.values())

            # Compute expval component
            expval, var = _expval_with_variance(dat, diagonal=diagonal)
            # Accumulate
            combined_expval += expval * coeff
            combined_variance += var * coeff ** 2
            combined_stderr += np.sqrt(max(var * coeff ** 2 / shots, 0.0))

        return ExpectationValueResult(
            combined_expval,
            combined_variance,
            [(combined_expval - combined_stderr, combined_expval + combined_stderr)],
        )


def _expval_with_variance(
    counts: Counts,
    diagonal: Optional[np.ndarray] = None,
    # clbits: Optional[list[int]] = None,
) -> tuple[float, float]:

    # Marginalize counts
    # if clbits is not None:
    #    counts = marginal_counts(counts, meas_qubits=clbits)

    # Get counts shots and probabilities
    probs = np.array(list(counts.values()))
    shots = probs.sum()
    probs = probs / shots

    # Get diagonal operator coefficients
    if diagonal is None:
        coeffs = np.array(
            [(-1) ** (key.count("1") % 2) for key in counts.keys()], dtype=probs.dtype
        )
    else:
        keys = [int(key, 2) for key in counts.keys()]
        coeffs = np.asarray(diagonal[keys], dtype=probs.dtype)

    # Compute expval
    expval = coeffs.dot(probs)

    # Compute variance
    if diagonal is None:
        # The square of the parity diagonal is the all 1 vector
        sq_expval = np.sum(probs)
    else:
        sq_expval = (coeffs ** 2).dot(probs)
    variance = sq_expval - expval ** 2

    # Compute standard deviation
    if variance < 0:
        if not np.isclose(variance, 0):
            logger.warning(
                "Encountered a negative variance in expectation value calculation."
                "(%f). Setting standard deviation of result to 0.",
                variance,
            )
        variance = 0.0
    return expval.item(), variance.item()


def _pauli_diagonal(pauli: str) -> np.ndarray:
    """Return diagonal for given Pauli.

    Args:
        pauli: a pauli string.

    Returns:
        np.ndarray: The diagonal vector for converting the Pauli basis
                    measurement into an expectation value.
    """
    if pauli[0] in ["+", "-"]:
        pauli = pauli[1:]

    diag = np.array([1])
    for i in reversed(pauli):
        if i == "I":
            tmp = np.array([1, 1])
        else:
            tmp = np.array([1, -1])
        diag = np.kron(tmp, diag)
    return diag

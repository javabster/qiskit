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

from typing import List, Optional, Tuple

import numpy as np

from qiskit import transpile
from qiskit.result import Counts

from .base_expectation_value import BaseExpectationValue
from .expectation_value_result import ExpectationValueResult


class PauliExpectationValue(BaseExpectationValue):
    def _preprocessing(self):
        # circuit transpilation
        transpiled_circuit = transpile(self._state, self._backend)  # TODO: option
        # TODO: final layout

        circuits = []
        metadata = []
        for pauli, coeff in self._observable.label_iter():
            circuit = transpiled_circuit.copy()
            for i, val in enumerate(reversed(pauli)):
                if val == "Y":
                    circuit.sdg(i)
                if val in ["Y", "X"]:
                    circuit.h(i)
            circuit.measure_all()
            circuits.append(circuit)
            metadata.append({"basis": pauli, "coeff": coeff})

        return circuits, metadata

    def _postprocessing(self, result):
        counts = result.get_counts()
        data = counts if isinstance(counts, list) else [counts]
        metadata = self._metadata

        combined_expval = 0.0
        combined_variance = 0.0
        combined_stderr = 0.0

        for dat, meta in zip(data, metadata):
            basis = meta.get("basis", None)
            diagonal = self._pauli_diagonal(basis) if basis is not None else None
            coeff = meta.get("coeff", 1)
            qubits = meta.get("qubits", None)
            shots = sum(dat.values())

            # Compute expval component
            expval, var = self._expval_with_variance(
                dat, diagonal=diagonal, mitigator=self._mitigator, mitigator_qubits=qubits
            )
            # Accumulate
            combined_expval += expval * coeff
            combined_variance += var * coeff ** 2
            combined_stderr += np.sqrt(max(var * coeff ** 2 / shots, 0.0))

        return ExpectationValueResult(
            np.array(combined_expval),
            np.array(combined_variance),
            [(combined_expval - combined_stderr, combined_expval + combined_stderr)],
        )

    @staticmethod
    def _expval_with_variance(
        counts: Counts,
        diagonal: Optional[np.ndarray] = None,
        clbits: Optional[List[int]] = None,
        mitigator: Optional = None,
        mitigator_qubits: Optional[List[int]] = None,
    ) -> Tuple[float, float]:
        if mitigator is not None:
            return mitigator.expectation_value(
                counts, diagonal=diagonal, clbits=clbits, qubits=mitigator_qubits
            )

        # Marginalize counts
        if clbits is not None:
            counts = marginal_counts(counts, meas_qubits=clbits)

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
        return expval, variance

    @staticmethod
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

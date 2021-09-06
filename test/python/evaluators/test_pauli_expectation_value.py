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

"""Tests for PauliExpectationValue."""

import unittest

from qiskit import BasicAer
from qiskit.circuit.library import RealAmplitudes
from qiskit.evaluators import PauliExpectationValue
from qiskit.opflow import PauliSumOp
from qiskit.test import QiskitTestCase
from qiskit.utils import has_aer

if has_aer():
    from qiskit.providers.aer import AerSimulator


class TestPauliExpectationValue(QiskitTestCase):
    """Test PauliExpectationValue"""

    def test_evaluate_basicaer(self):
        """test for evaluate with BasicAer"""
        observable = PauliSumOp.from_list([("XX", 1), ("YY", 2), ("ZZ", 3)])
        ansatz = RealAmplitudes(num_qubits=2, reps=2)
        expval = PauliExpectationValue(
            ansatz, observable, backend=BasicAer.get_backend("qasm_simulator")
        )
        expval.set_transpile_options(seed_transpiler=15)
        expval.set_run_options(seed_simulator=15)
        result = expval.evaluate([0, 1, 1, 2, 3, 5], had_transpiled=False)
        self.assertAlmostEqual(result.value, 1.845703125)
        self.assertAlmostEqual(result.variance, 11.117565155029297)

    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    def test_evaluate(self):
        """test for evaluate with Aer"""
        observable = PauliSumOp.from_list([("XX", 1), ("YY", 2), ("ZZ", 3)])
        ansatz = RealAmplitudes(num_qubits=2, reps=2)
        expval = PauliExpectationValue(ansatz, observable, backend=AerSimulator())
        expval.set_transpile_options(seed_transpiler=15)
        expval.set_run_options(seed_simulator=15)
        result = expval.evaluate([0, 1, 1, 2, 3, 5])
        self.assertAlmostEqual(result.value, 1.806640625)
        self.assertAlmostEqual(result.variance, 11.092403411865234)

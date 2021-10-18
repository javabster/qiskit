# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Test QuantumCircuit.compose()."""

import unittest

from qiskit import transpile
from qiskit.pulse import Schedule
from qiskit.circuit import (
    QuantumRegister,
    ClassicalRegister,
    QuantumCircuit,
    Parameter,
    Gate,
    Instruction,
)
from qiskit.circuit.library import HGate, RZGate, CXGate, CCXGate
from qiskit.test import QiskitTestCase


class TestCircuitCompose(QiskitTestCase):
    """Test composition of two circuits."""

    def setUp(self):
        super().setUp()
        qreg1 = QuantumRegister(3, "lqr_1")
        qreg2 = QuantumRegister(2, "lqr_2")
        creg = ClassicalRegister(2, "lcr")

        self.circuit_left = QuantumCircuit(qreg1, qreg2, creg)
        self.circuit_left.h(qreg1[0])
        self.circuit_left.x(qreg1[1])
        self.circuit_left.p(0.1, qreg1[2])
        self.circuit_left.cx(qreg2[0], qreg2[1])

        self.left_qubit0 = qreg1[0]
        self.left_qubit1 = qreg1[1]
        self.left_qubit2 = qreg1[2]
        self.left_qubit3 = qreg2[0]
        self.left_qubit4 = qreg2[1]
        self.left_clbit0 = creg[0]
        self.left_clbit1 = creg[1]
        self.condition = (creg, 3)

    def test_wrapping_unitary_circuit(self):
        """Test a unitary circuit will be wrapped as Gate, else as Instruction."""
        qc_init = QuantumCircuit(1)
        qc_init.x(0)

        qc_unitary = QuantumCircuit(1, name="a")
        qc_unitary.ry(0.23, 0)

        qc_nonunitary = QuantumCircuit(1)
        qc_nonunitary.reset(0)

        with self.subTest("wrapping a unitary circuit"):
            qc = qc_init.compose(qc_unitary, wrap=True)
            self.assertIsInstance(qc.data[1][0], Gate)

        with self.subTest("wrapping a non-unitary circuit"):
            qc = qc_init.compose(qc_nonunitary, wrap=True)
            self.assertIsInstance(qc.data[1][0], Instruction)


if __name__ == "__main__":
    unittest.main()

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
Base Preprocessing class
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

from qiskit import QuantumCircuit
from qiskit.providers import Options
from qiskit.quantum_info import SparsePauliOp


class BasePreprocessing(ABC):
    """
    Base class of pre processing.
    """

    _default_transpile_options = Options()

    def __call__(self, state: QuantumCircuit, observable: SparsePauliOp):
        return self.execute(state, observable)

    @abstractmethod
    def execute(
        self, state: QuantumCircuit, observable: SparsePauliOp
    ) -> Union[list[QuantumCircuit], tuple[QuantumCircuit, list[QuantumCircuit]]]:
        """
        TODO
        """
        return NotImplemented

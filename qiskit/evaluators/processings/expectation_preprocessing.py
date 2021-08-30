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
Expectation Preprocessing class
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.providers import BackendV1 as Backend
from qiskit.providers import Options
from qiskit.quantum_info import SparsePauliOp

from .base_preprocessing import BasePreprocessing


class ExpectationPreprocessing(BasePreprocessing, ABC):
    _default_transpile_options = Options()

    def __init__(
        self,
        backend: Backend,
        transpile_options: Optional[dict] = None,
    ):
        self._backend = backend
        self._transpile_options = self._default_transpile_options
        if transpile_options is not None:
            self.set_transpile_options(**transpile_options)

    @property
    def transpile_options(self) -> Options:
        """Return the transpiler options for transpiling the circuits."""
        return self._transpile_options

    def set_transpile_options(self, **fields) -> ExpectationPreprocessing:
        """Set the transpiler options for transpiler.
        Args:
            fields: The fields to update the options
        """
        self._transpile_options.update_options(**fields)
        return self

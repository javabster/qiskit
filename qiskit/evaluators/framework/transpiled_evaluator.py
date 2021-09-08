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
Evaluator class base class
"""
from __future__ import annotations

from typing import Union

from qiskit import QuantumCircuit
from qiskit.evaluators.backends import BaseBackendWrapper, ShotBackendWrapper
from qiskit.providers import BackendV1 as Backend

from .base_evaluator import BaseEvaluator, Postprocessing


class TranspiledEvaluator(BaseEvaluator):
    """
    Evaluator for transpiled circuits.
    """

    def __init__(
        self,
        transpiled_circuits: list[QuantumCircuit],
        postprocessing: Postprocessing,
        backend: Union[Backend, BaseBackendWrapper, ShotBackendWrapper],
    ):
        """
        Args:
            backend: backend
        """
        super().__init__(backend=backend, postprocessing=postprocessing)
        self._transpiled_circuits = transpiled_circuits

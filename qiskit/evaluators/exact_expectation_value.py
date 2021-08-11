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


class ExactExpectationValue(BaseExpectationValue):
    def _preprocessing(self):
        # circuit transpilation
        transpiled_circuit = transpile(self._state, self._backend)  # TODO: option
        # TODO: final layout

        transpiled_circuit.save_expectation_value_variance(
            operator=self._observable, qubits=range(transpiled_circuit.num_qubits)
        )

        return [transpiled_circuit], [{}]

    def _postprocessing(self, result):
        expval, variance = result.data()["expectation_value_variance"]

        return ExpectationValueResult(
            np.array(expval),
            np.array(variance),
            None,
        )

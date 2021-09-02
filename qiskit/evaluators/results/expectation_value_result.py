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
Expectation value result class
"""

from __future__ import annotations

import numpy as np

from .base_result import BaseResult


class ExpectationValueResult(BaseResult):
    """
    Result of ExpectationValue
    #TODO doc
    """

    value: np.ndarray
    variance: np.ndarray
    confidence_interval: list[tuple[float, float]]
    # metadata: Metadata

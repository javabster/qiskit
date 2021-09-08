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
Expectation value array result class
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base_result import BaseResult


class ExpectationValueArrayResult(BaseResult):
    """
    Result of ExpectationValue
    #TODO doc
    """

    values: np.ndarray[Any, np.dtype[np.float64]]
    variances: np.ndarray[Any, np.dtype[np.float64]]
    confidence_intervals: np.ndarray[Any, np.dtype[np.float64]]
    # metadata: Metadata

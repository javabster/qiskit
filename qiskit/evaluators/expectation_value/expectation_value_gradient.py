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
Expectation value gradient class
"""

from __future__ import annotations

from typing import Union, cast

import numpy as np

from ..results.expectation_value_gradient_result import ExpectationValueGradientResult
from .expectation_value import ExpectationValue


class BaseExpectationValueGradient:
    """
    Expectation Value Gradient class
    """

    def __init__(self, expval: ExpectationValue):
        self._expval = expval

    def evaluate(
        self, parameters: Union[list[float], np.ndarray], **run_options
    ) -> ExpectationValueGradientResult:
        return NotImplemented


class FiniteDiffGradient(BaseExpectationValueGradient):
    """
    Finite difference of expectation values
    """

    def __init__(self, expval: ExpectationValue, epsilon: float):
        super(FiniteDiffGradient, self).__init__(expval)
        self._epsilon = epsilon

    def evaluate_seq(
        self,
        parameters: Union[list[float], np.ndarray],
        **run_options,
    ) -> ExpectationValueGradientResult:
        if isinstance(parameters, np.ndarray):
            if len(parameters.shape) != 1:
                raise ValueError("parameters should be a vector")
            else:
                parameters = parameters.tolist()
        parameters = cast(list[float], parameters)
        dim = len(parameters)
        todo = [parameters]

        # preprocessing
        for i in range(dim):
            ei = parameters.copy()
            ei[i] += self._epsilon
            todo.append(ei)

        # execution
        results = []
        for param in todo:
            results.append(self._expval.evaluate(param, **run_options))

        # postprocessing
        grad = np.zeros(dim)
        f_ref = results[0].value
        for i, result in enumerate(results[1:]):
            f_i = result.value
            grad[i] = (f_i - f_ref) / self._epsilon

        return ExpectationValueGradientResult(values=grad)

    def evaluate(
        self,
        parameters: Union[list[float], np.ndarray],
        **run_options,
    ) -> ExpectationValueGradientResult:
        if isinstance(parameters, np.ndarray):
            if len(parameters.shape) != 1:
                raise ValueError("parameters should be a vector")
            else:
                parameters = parameters.tolist()
        parameters = cast(list[float], parameters)
        dim = len(parameters)
        todo = [parameters]

        # preprocessing
        for i in range(dim):
            ei = parameters.copy()
            ei[i] += self._epsilon
            todo.append(ei)

        # execution
        results = self._expval.evaluate(todo, **run_options)

        # postprocessing
        grad = np.zeros(dim)
        f_ref = results.items[0].value
        for i, result in enumerate(results.items[1:]):
            f_i = result.value
            grad[i] = (f_i - f_ref) / self._epsilon

        return ExpectationValueGradientResult(values=grad)


class ParameterShiftGradient(BaseExpectationValueGradient):
    """
    Gradient of expectation values by parameter shift
    """

    def __init__(self, expval: ExpectationValue):
        super(ParameterShiftGradient, self).__init__(expval)

    def evaluate(
        self,
        parameters: Union[list[float], np.ndarray],
        **run_options,
    ) -> ExpectationValueGradientResult:
        if isinstance(parameters, np.ndarray):
            if len(parameters.shape) != 1:
                raise ValueError("parameters should be a vector")
            else:
                parameters = parameters.tolist()
        parameters = cast(list[float], parameters)
        dim = len(parameters)
        todo = []
        epsilon = np.pi / 2

        # preprocessing
        for i in range(dim):
            ei = parameters.copy()
            ei[i] += epsilon
            todo.append(ei)

            ei = parameters.copy()
            ei[i] -= epsilon
            todo.append(ei)

        # execution
        results = self._expval.evaluate(todo, **run_options)

        # postprocessing
        div = 2 * np.sin(epsilon)
        grad = np.zeros(dim)
        for i in range(dim):
            f_plus = results.items[2 * i].value
            f_minus = results.items[2 * i + 1].value
            grad[i] = (f_plus - f_minus) / div

        return ExpectationValueGradientResult(values=grad)

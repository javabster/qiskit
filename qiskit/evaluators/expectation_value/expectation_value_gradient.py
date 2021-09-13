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

from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import numpy as np

#from qiskit.evaluators.framework.base_evaluator import BaseEvaluator
from qiskit.evaluators.results import (
    ExpectationValueArrayResult,
    ExpectationValueGradientResult,
)

from .expectation_value import ExpectationValue


class BaseExpectationValueGradient(ABC):  # (BaseEvaluator):
    """
    Base class for expectation value gradient
    """

    def __init__(self, expval: ExpectationValue):
        self._expval = expval

    @abstractmethod
    def _preprocessing(
        self, parameters: "np.ndarray[Any, np.dtype[np.float64]]"
    ) -> "np.ndarray[Any, np.dtype[np.float64]]":
        return NotImplemented

    @abstractmethod
    def _postprocessing(
        self, results: ExpectationValueArrayResult, shape: Union[Tuple[int], Tuple[int, int]]
    ) -> ExpectationValueGradientResult:
        return NotImplemented

    def evaluate(
        self,
        parameters: Union[list[float], list[list[float]], np.ndarray[Any, np.dtype[np.float64]]],
        **run_options,
    ) -> ExpectationValueGradientResult:
        """TODO
        """
        parameters = np.asarray(parameters, dtype=np.float64)
        if len(parameters.shape) not in [1, 2]:
            raise ValueError("parameters should be a 1D vector or 2D vectors")
        param_array = self._preprocessing(parameters)
        results = self._expval.evaluate(param_array, **run_options)
        return self._postprocessing(results, parameters.shape)


class FiniteDiffGradient(BaseExpectationValueGradient):
    """
    Finite difference of expectation values
    """

    def __init__(self, expval: ExpectationValue, epsilon: float):
        super().__init__(expval)
        self._epsilon = epsilon

    def _preprocessing(
        self, parameters: "np.ndarray[Any, np.dtype[np.float64]]"
    ) -> "np.ndarray[Any, np.dtype[np.float64]]":
        if len(parameters.shape) == 1:
            parameters = parameters.reshape((1, parameters.shape[0]))
        dim = parameters.shape[-1]
        ret = []
        for param in parameters:
            ret.append(param)
            for i in range(dim):
                ei = param.copy()
                ei[i] += self._epsilon
                ret.append(ei)
        return np.array(ret)

    def _postprocessing(
        self, results: ExpectationValueArrayResult, shape: Union[Tuple[int], Tuple[int, int]]
    ) -> ExpectationValueGradientResult:
        dim = shape[-1]
        array = results.values.reshape((results.values.shape[0] // (dim + 1), dim + 1))
        ret = []
        for values in array:
            grad = np.zeros(dim)
            f_ref = values[0]
            for i, f_i in enumerate(values[1:]):
                grad[i] = (f_i - f_ref) / self._epsilon
            ret.append(grad)
        grad = np.array(ret).reshape(shape)
        return ExpectationValueGradientResult(values=grad)


class ParameterShiftGradient(BaseExpectationValueGradient):
    """
    Gradient of expectation values by parameter shift
    """

    def __init__(self, expval: ExpectationValue):
        super().__init__(expval)
        self._epsilon = np.pi / 2

    def _preprocessing(
        self, parameters: "np.ndarray[Any, np.dtype[np.float64]]"
    ) -> "np.ndarray[Any, np.dtype[np.float64]]":
        if len(parameters.shape) == 1:
            parameters = parameters.reshape((1, parameters.shape[0]))
        dim = parameters.shape[-1]
        ret = []
        for param in parameters:
            for i in range(dim):
                ei = param.copy()
                ei[i] += self._epsilon
                ret.append(ei)

                ei = param.copy()
                ei[i] -= self._epsilon
                ret.append(ei)

        return np.array(ret)

    def _postprocessing(
        self, results: ExpectationValueArrayResult, shape: Union[Tuple[int], Tuple[int, int]]
    ) -> ExpectationValueGradientResult:
        dim = shape[-1]
        array = results.values.reshape((results.values.shape[0] // (2 * dim), 2 * dim))
        div = 2 * np.sin(self._epsilon)
        ret = []
        for values in array:
            grad = np.zeros(dim)
            for i in range(dim):
                f_plus = values[2 * i]
                f_minus = values[2 * i + 1]
                grad[i] = (f_plus - f_minus) / div
            ret.append(grad)
        grad = np.array(ret).reshape(shape)
        return ExpectationValueGradientResult(values=grad)

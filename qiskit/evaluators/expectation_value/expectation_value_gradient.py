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
from typing import Any, Optional, Union, cast

import numpy as np

from qiskit import QuantumCircuit
from qiskit.evaluators.framework.base_evaluator import BaseEvaluator
from qiskit.evaluators.results import (
    CompositeResult,
    ExpectationValueGradientResult,
)

from .expectation_value import ExpectationValue


class BaseExpectationValueGradient(BaseEvaluator, ABC):
    """
    Base class for expectation value gradient
    """

    def __init__(
        self,
        expectation_value: ExpectationValue,
        append: bool = False,
    ):
        self._expectation_value = expectation_value
        super().__init__(
            backend=self._expectation_value._backend,
            postprocessing=self._expectation_value._postprocessing,
            transpile_options=self._expectation_value.transpile_options.__dict__,
            append=append,
        )

    @property
    def preprocessed_circuits(self) -> list[QuantumCircuit]:
        """
        Preprocessed quantum circuits produced by preprocessing

        Returns:
            List of the transpiled quantum circuit
        """
        return self._expectation_value.preprocessed_circuits

    @property
    def transpiled_circuits(self) -> list[QuantumCircuit]:
        """
        Transpiled quantum circuits.

        Returns:
            List of the transpiled quantum circuit
        """
        return self._expectation_value.transpiled_circuits

    @abstractmethod
    def _eval_parameters(
        self, parameters: "np.ndarray[Any, np.dtype[np.float64]]"
    ) -> "np.ndarray[Any, np.dtype[np.float64]]":
        return NotImplemented

    @abstractmethod
    def _compute_gradient(self, results: CompositeResult, shape) -> ExpectationValueGradientResult:
        return NotImplemented

    def evaluate(
        self,
        parameters: Optional[
            Union[list[float], list[list[float]], np.ndarray[Any, np.dtype[np.float64]]]
        ] = None,
        **run_options,
    ) -> ExpectationValueGradientResult:
        """TODO"""
        if parameters is None:
            raise ValueError()

        parameters = np.asarray(parameters, dtype=np.float64)
        if len(parameters.shape) not in [1, 2]:
            raise ValueError("parameters should be a 1D vector or 2D vectors")
        param_array = self._eval_parameters(parameters)
        results = cast(CompositeResult, super().evaluate(param_array, **run_options))
        return self._compute_gradient(results, parameters.shape)


class FiniteDiffGradient(BaseExpectationValueGradient):
    """
    Finite difference of expectation values
    """

    def __init__(self, expectation_value: ExpectationValue, epsilon: float):
        super().__init__(expectation_value)
        self._epsilon = epsilon

    def _eval_parameters(
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

    def _compute_gradient(self, results: CompositeResult, shape) -> ExpectationValueGradientResult:
        values = np.array([r.value for r in results.items])  # type: ignore
        dim = shape[-1]
        array = values.reshape((values.shape[0] // (dim + 1), dim + 1))
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

    def __init__(self, expectation_value: ExpectationValue):
        super().__init__(expectation_value)
        self._epsilon = np.pi / 2

    def _eval_parameters(
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

    def _compute_gradient(self, results: CompositeResult, shape) -> ExpectationValueGradientResult:
        values = np.array([r.value for r in results.items])  # type: ignore
        dim = shape[-1]
        array = values.reshape((values.shape[0] // (2 * dim), 2 * dim))
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

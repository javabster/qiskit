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
Joint evaluator class
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from qiskit.evaluators.backends import ShotResult
from qiskit.evaluators.framework.base_evaluator import BaseEvaluator
from qiskit.evaluators.results import CompositeResult
from qiskit.evaluators.results.base_result import BaseResult

if TYPE_CHECKING:
    from typing import Any


class JointEvaluator(BaseEvaluator):
    """Joint Evaluator"""

    def __init__(self, evaluators: list[BaseEvaluator]):
        """hoge"""

        self._evaluators = evaluators

        self._counter = 0
        super().__init__(evaluators[0]._backend, self._construct_postprocessing)

        for evaluator in evaluators:
            if evaluator.backend != self.backend:
                raise ValueError("")
            # Should we update the run_options?
            # self.run_options.update_options(**evaluator.run_options.__dict__)

    @property
    def transpiled_circuits(self):
        if self._transpiled_circuits is None:
            self._transpiled_circuits = sum(
                [evaluator.transpiled_circuits for evaluator in self._evaluators], []
            )
        return self._transpiled_circuits

    def _construct_postprocessing(self, result: Union[dict, ShotResult]) -> BaseResult:
        current_counter = self._counter
        self._counter += 1
        if self._counter == len(self._evaluators):
            self._counter = 0
        return self._evaluators[current_counter]._postprocessing(result)

    def evaluate(
        self,
        parameters: Optional[
            Union[
                list[float],
                list[list[float]],
                "np.ndarray[Any, np.dtype[np.float64]]",
            ]
        ] = None,
        **run_options,
    ) -> CompositeResult:
        run_opts = copy.copy(self.run_options)
        run_opts.update_options(**run_options)
        run_opts_dict = run_opts.__dict__

        if len(parameters) != len(self._evaluators):
            raise TypeError("Length is different.")

        if parameters is None:
            circuits = self.transpiled_circuits
        else:
            parameters = np.asarray(parameters, dtype=np.float64)
            if parameters.ndim == 2:
                circuits = [
                    circ.bind_parameters(param)
                    for param in parameters
                    for evaluator in self._evaluators
                    for circ in evaluator.transpiled_circuits
                ]
            elif parameters.ndim == 1:
                circuits = [
                    circ.bind_parameters(parameters)  # type: ignore
                    for evaluator in self._evaluators
                    for circ in evaluator.transpiled_circuits
                ]

        results = self._backend.run_and_wait(circuits, **run_opts_dict)

        accum = 0
        postprocessed = []
        for evaluator in self._evaluators:
            postprocessed.append(
                self._postprocessing(results[accum : accum + len(evaluator.preprocessed_circuits)])
            )

        return CompositeResult(postprocessed)

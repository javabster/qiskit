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

from typing import Optional, Union

from qiskit.evaluators.backends import ShotResult
from qiskit.evaluators.framework.base_evaluator import BaseEvaluator
from qiskit.evaluators.results.base_result import BaseResult
from qiskit.result import Result


class JointEvaluator(BaseEvaluator):
    """Joint Evaluator"""

    def __init__(self, evaluators: list[BaseEvaluator]):
        """hoge"""

        self._evaluators = evaluators
        self._num_evaluators = len(evaluators)
        self._num_circuits: Optional[list[int]]

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
            transpiled_circuits_list = [
                evaluator.transpiled_circuits for evaluator in self._evaluators
            ]
            metadata_list = [evaluator._metadata for evaluator in self._evaluators]
            self._transpiled_circuits = sum(transpiled_circuits_list, [])
            self._num_circuits = [len(circuits) for circuits in transpiled_circuits_list]
            self._metadata = sum(metadata_list, [])
        return self._transpiled_circuits

    def _construct_postprocessing(
        self, result: Union[Result, ShotResult], metadata: list[dict]
    ) -> BaseResult:
        current_counter = self._counter
        self._counter += 1
        if self._counter == self._num_evaluators:
            self._counter = 0
        return self._evaluators[current_counter]._postprocessing(result, metadata)

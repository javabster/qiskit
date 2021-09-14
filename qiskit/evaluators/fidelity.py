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
Fidelity
"""
from .framework.base_evaluator import BaseEvaluator


class Fidelity(BaseEvaluator):
    """
    The class evaluates the fidelity of two states.
    """

    # pylint: disable=unused-argument
    def __init__(self, state1, state2, backend, postprocessing):
        """
        TODO: write docstring
        """
        super().__init__(backend, postprocessing)
        pass

    def evaluate(self, parameters=None, **run_options):
        """
        TODO
        """
        pass

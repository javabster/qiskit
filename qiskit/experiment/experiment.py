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
Base Experiment class.
"""

from abc import ABC, abstractmethod
from typing import Optional, List

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers import BaseJob
from qiskit.result import Result
from qiskit.exceptions import QiskitError
from qiskit.providers import BaseBackend


class Experiment(ABC):
    """Base experiment class."""

    # pylint: disable=arguments-differ
    def __init__(self, num_qubits: int):
        """Initialize an experiment."""
        # Circuit generation parameters
        self._num_qubits = num_qubits

        # Experiment Jobs
        self._jobs = []

    @property
    def job(self):
        """Return the last submitted job"""
        if not self._jobs:
            raise QiskitError("No experiment job has been submitted.")
        return self._jobs[-1]

    @property
    def jobs(self):
        """Return the last submitted job"""
        return self._jobs

    @property
    def num_qubits(self):
        """Return the number of qubits for this experiment."""
        return self._num_qubits

    def execute(self,
                backend: BaseBackend,
                qubits: Optional[List[int]] = None,
                **kwargs,
                ) -> BaseJob:
        """Execute the experiment on a backend.
​
        TODO: Add transpiler, schedule, assembler options for backend here

        Args:
            backend: backend to run experiment on.
            qubits: Optional, apply the N-qubit calibration circuits to
                    these device qubits.
            kwargs: kwargs for assemble method.
​
        Returns:
            BaseJob: the experiment job.
        """
        if qubits or backend:
            initial_layout = qubits
        else:
            initial_layout = None
        circuits = transpile(self.circuits(),
                             backend=backend,
                             initial_layout=initial_layout)
        if qubits is None:
            metadata = self.metadata()
        else:
            metadata = []
            for meta in self.metadata:
                new_meta = meta.copy()
                new_meta['qubits'] = qubits
            metadata.append(new_meta)
        qobj = assemble(circuits,
                        backend=backend,
                        qobj_header={'metadata': metadata},
                        **kwargs)
        job = backend.run(qobj)
        self._jobs.append(job)
        return job

    @abstractmethod
    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits."""

    @abstractmethod
    def metadata(self) -> List[dict]:
        """Generate a list of experiment metadata dicts."""


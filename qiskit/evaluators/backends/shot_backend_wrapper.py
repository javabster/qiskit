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
Shot Backend wrapper class
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Union

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import MissingOptionalLibraryError, QiskitError
from qiskit.providers.backend import BackendV1
from qiskit.result import Counts, Result

from .backend_wrapper import BackendWrapper, BaseBackendWrapper, ReadoutErrorMitigation

logger = logging.getLogger(__name__)


@dataclass
class ShotResult:
    """
    Dataclass for shot results
    """

    counts: list[Counts]
    shots: int
    raw_results: list[Result]
    metadata: list[dict]

    def __getitem__(self, key):
        return ShotResult(self.counts[key], self.shots, self.raw_results, self.metadata[key])


class ShotBackendWrapper(BaseBackendWrapper[ShotResult]):
    """Backend wrapper to return a list of counts"""

    def __init__(self, backend: Union[BackendV1, BaseBackendWrapper]):
        self._backend = BackendWrapper.from_backend(backend)

        config = self._backend.backend.configuration()
        self._max_shots = config.max_shots
        if hasattr(config, "max_experiments"):
            self._max_experiments = config.max_experiments
        else:
            self._max_experiments = 1 if self.is_aer(self._backend.backend) else 1_000_000
            logger.warning(
                "no max_experiments for backend '%s'. Set %d as max_experiments.",
                self._backend.backend.name(),
                self._max_experiments,
            )
        self._num_circuits = 0
        self._num_splits = 0
        self._raw_results: list[Result] = []

    @staticmethod
    def is_aer(backend: BackendV1):
        """
        Returns True if the backend is a subclass of AerProvider.
        """
        try:
            from qiskit.providers.aer import AerProvider

            return isinstance(backend.provider(), AerProvider)
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="qiskit-aer",
                name="Qiskit Aer",
                pip_install="pip install qiskit-aer",
            ) from ex

    @property
    def backend(self) -> BackendV1:
        """
        TODO

        Returns:
            backend
        """
        return self._backend.backend

    @property
    def max_shots(self) -> int:
        """
        TODO

        Returns:
            max_shots
        """
        return self._max_shots

    @property
    def max_experiments(self) -> int:
        """
        TODO

        Returns:
            max_experiments
        """
        return self._max_experiments

    @staticmethod
    def from_backend(
        backend: Union[BackendV1, BaseBackendWrapper, ShotBackendWrapper]
    ) -> ShotBackendWrapper:
        """
        Backend to ShotBackendWrapper

        Returns:
            wrapped backend
        """
        if isinstance(backend, (BackendV1, BaseBackendWrapper)):
            return ShotBackendWrapper(backend)
        return backend

    def _split_experiments(
        self, circuits: list[QuantumCircuit], shots: int
    ) -> list[tuple[list[QuantumCircuit], int]]:
        assert self._num_circuits > self._max_experiments
        ret = []
        remaining_shots = shots
        splits = []
        for i in range(0, self._num_circuits, self._max_experiments):
            splits.append(circuits[i : min(i + self._max_experiments, self._num_circuits)])
        self._num_splits = len(splits)
        logger.info("Number of circuits %d, Number of splits: %d", len(circuits), self._num_splits)
        while remaining_shots > 0:
            shots = min(remaining_shots, self._max_shots)
            remaining_shots -= shots
            for circs in splits:
                ret.append((circs, shots))
        return ret

    def _copy_experiments(
        self, circuits: list[QuantumCircuit], shots: int, exact: bool
    ) -> list[tuple[list[QuantumCircuit], int]]:
        assert self._num_circuits <= self._max_experiments
        max_copies = self._max_experiments // self._num_circuits
        ret = []
        remaining_shots = shots
        while remaining_shots > 0:
            num_copies, rem = divmod(remaining_shots, self._max_shots)
            if rem:
                num_copies += 1
            num_copies = min(num_copies, max_copies)

            shots, rem = divmod(remaining_shots, num_copies)
            if rem and not exact:
                shots += 1
            shots = min(shots, self._max_shots)
            logger.info(
                "Number of circuits %d, number of shots: %d, number of copies: %d, "
                "total number of shots: %d",
                len(circuits),
                shots,
                num_copies,
                shots * num_copies,
            )
            remaining_shots -= shots * num_copies
            ret.append((circuits * num_copies, shots))
        return ret

    # pylint: disable=arguments-differ
    def run_and_wait(
        self,
        circuits: Union[QuantumCircuit, list[QuantumCircuit]],
        append: bool = False,
        exact_shots: bool = False,
        **options,
    ) -> ShotResult:
        """
        TODO

        Returns:
            list of counts
        """
        if "shots" in options:
            shots = options["shots"]
            del options["shots"]
        else:
            shots = self._backend.backend.options.shots
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        self._num_circuits = len(circuits)
        if self._num_circuits > self._max_experiments:
            circs_shots = self._split_experiments(circuits, shots)
        else:
            circs_shots = self._copy_experiments(circuits, shots, exact_shots)
        results = []
        for circs, shots in circs_shots:
            result = self._backend.run_and_wait(circs, shots=shots, **options)
            results.append(result)

        if isinstance(self._backend, ReadoutErrorMitigation):
            results = self._backend.apply_mitigation(results)
        if append:
            self._raw_results.extend(results)
        else:
            self._raw_results = results
        counts = self._get_counts(self._raw_results)
        metadata = [res.header.metadata for result in results for res in result.results]
        return ShotResult(
            counts=counts,
            shots=int(sum(counts[0].values())),
            raw_results=self._raw_results,
            metadata=metadata,
        )

    def _get_counts(self, results: list[Result]) -> list[Counts]:
        """
        Convert Result to Counts

        Returns:
            list of counts
        Raises:
            QiskitError: if inputs are empty
        """
        if len(results) == 0:
            raise QiskitError("Empty result")
        counters: list[Counter] = [Counter() for _ in range(self._num_circuits)]
        i = 0
        for result in results:
            counts = result.get_counts()
            if isinstance(counts, Counts):
                counts = [counts]
            for count in counts:
                counters[i % self._num_circuits].update(count)
                i += 1
        # TODO: recover the metadata of Counts
        return [Counts(c) for c in counters]

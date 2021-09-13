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
Backend wrapper classes
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Generic, TypeVar, Union

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.providers.backend import BackendV1
from qiskit.result import Counts, Result

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .shot_backend_wrapper import ShotBackendWrapper

T = TypeVar("T")  # pylint: disable=invalid-name


class BaseBackendWrapper(ABC, Generic[T]):
    """
    TODO
    """

    @abstractmethod
    def run_and_wait(self, circuits: Union[QuantumCircuit, list[QuantumCircuit]], **options) -> T:
        """
        TODO
        """
        return NotImplemented

    @property
    @abstractmethod
    def backend(self) -> BackendV1:
        """
        TODO
        """
        return NotImplemented


class BackendWrapper(BaseBackendWrapper[Result]):
    """
    TODO
    """

    def __init__(self, backend: BackendV1):
        """
        TODO
        """
        self._backend = backend

    @property
    def backend(self) -> BackendV1:
        """
        TODO
        """
        return self._backend

    def run_and_wait(
        self, circuits: Union[QuantumCircuit, list[QuantumCircuit]], **options
    ) -> Result:
        """
        TODO
        """
        job = self._backend.run(circuits, **options)
        return job.result()

    @classmethod
    def from_backend(cls, backend: Union[BackendV1, BaseBackendWrapper]) -> BaseBackendWrapper:
        """
        TODO
        """
        if isinstance(backend, BackendV1):
            return cls(backend)
        return backend

    @staticmethod
    def to_backend(backend: Union[BackendV1, BaseBackendWrapper, ShotBackendWrapper]) -> BackendV1:
        """
        TODO
        """
        if isinstance(backend, BackendV1):
            return backend
        return backend.backend


class Retry(BaseBackendWrapper):
    """
    TODO
    """

    def __init__(self, backend: BackendV1):
        """
        TODO
        """
        self._backend = backend

    @property
    def backend(self):
        """
        TODO
        """
        return self._backend

    @staticmethod
    def _get_result(job):
        """Get a result of a job. Will retry when ``IBMQJobApiError`` (i.e., network error)

        ``IBMQJob.result`` raises the following errors.
            - IBMQJobInvalidStateError: If the job was cancelled.
            - IBMQJobFailureError: If the job failed.
            - IBMQJobApiError: If an unexpected error occurred when communicating with the server.
        """
        try:
            from qiskit.providers.ibmq.job import IBMQJobApiError
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="qiskit-ibmq-provider",
                name="IBMQ Provider",
                pip_install="pip install qiskit-ibmq-provider",
            ) from ex

        while True:
            try:
                return job.result()
            except IBMQJobApiError as ex:  # network error, will retry to get a result
                logger.warning(ex.message)

    def run_and_wait(
        self, circuits: Union[QuantumCircuit, list[QuantumCircuit]], **options
    ) -> Result:
        """
        TODO
        """
        try:
            from qiskit.providers.ibmq.job import IBMQJobFailureError, IBMQJobInvalidStateError
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="qiskit-ibmq-provider",
                name="IBMQ Provider",
                pip_install="pip install qiskit-ibmq-provider",
            ) from ex

        while True:
            job = self._backend.run(circuits, **options)
            try:
                result = self._get_result(job)
            except IBMQJobInvalidStateError as ex:  # cancelled, will retry to submit a job
                logger.warning(ex.message)
                logger.info("Job was cancelled %s. Retry another job.", job.job_id())
                continue
            except IBMQJobFailureError as ex:  # job failed, will terminate
                logger.warning(ex.message)
                raise ex

            if result.success:
                return result
            else:
                logger.warning("job finished unsuccessfully %s", job.job_id())


class ReadoutErrorMitigation(BaseBackendWrapper):
    """
    TODO
    """

    # need to move to the new mitigator class in the future
    # https://github.com/Qiskit/qiskit-terra/pull/6485
    # need to support M3 https://github.com/Qiskit-Partners/mthree
    # need to use Terra's error mitigation after merge of
    # https://github.com/Qiskit/qiskit-terra/pull/6867
    def __init__(
        self,
        backend: Union[BackendV1, BaseBackendWrapper],
        mitigation: str,
        refresh: float,
        shots: int,
        **cal_options,
    ):
        """
        TODO
        """
        self._backend = BackendWrapper.from_backend(backend)
        self._mitigation = mitigation
        self._refresh = timedelta(seconds=refresh)
        self._shots = shots
        self._time_threshold = datetime.fromordinal(1)
        self._cal_options = cal_options

        try:
            from qiskit.ignis.mitigation.measurement import CompleteMeasFitter, TensoredMeasFitter
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="qiskit-ignis",
                name="execute",
                pip_install="pip install qiskit-ignis",
            ) from ex
        try:
            from mthree import M3Mitigation
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                libname="mthree",
                name="execute",
                pip_install="pip install mthree",
            ) from ex
        self._meas_fitter: dict[
            datetime, Union[CompleteMeasFitter, TensoredMeasFitter, M3Mitigation]
        ] = {}

    @property
    def backend(self):
        """
        TODO
        """
        if isinstance(self._backend, BaseBackendWrapper):
            return self._backend.backend
        return self._backend

    @property
    def mitigation(self):
        """
        TODO
        """
        return self._mitigation

    @property
    def refresh(self):
        """
        TODO
        """
        return self._refresh

    @property
    def cal_options(self):
        """
        TODO
        """
        return self._cal_options

    @property
    def shots(self):
        """
        TODO
        """
        return self._shots

    @staticmethod
    def _datetime(data):
        """
        TODO
        """
        # Aer's result.data is str, but IBMQ's result.data is datetime
        if isinstance(data, str):
            return datetime.fromisoformat(data)
        return data

    def _maybe_calibrate(self):
        now = datetime.now()
        if now <= self._time_threshold:
            return
        logger.info("readout error mitigation calibration %s at %s", self._mitigation, now)
        if self._mitigation == "tensored":
            try:
                from qiskit.ignis.mitigation.measurement import (
                    TensoredMeasFitter,
                    tensored_meas_cal,
                )
            except ImportError as ex:
                raise MissingOptionalLibraryError(
                    libname="qiskit-ignis",
                    name="execute",
                    pip_install="pip install qiskit-ignis",
                ) from ex
            meas_calibs, state_labels = tensored_meas_cal(**self._cal_options)
            cal_results = self._backend.run_and_wait(meas_calibs, shots=self._shots)
            self._meas_fitter[now] = TensoredMeasFitter(cal_results, **self._cal_options)
        elif self._mitigation == "complete":
            try:
                from qiskit.ignis.mitigation.measurement import (
                    CompleteMeasFitter,
                    complete_meas_cal,
                )
            except ImportError as ex:
                raise MissingOptionalLibraryError(
                    libname="qiskit-ignis",
                    name="execute",
                    pip_install="pip install qiskit-ignis",
                ) from ex
            meas_calibs, state_labels = complete_meas_cal(**self._cal_options)
            cal_results = self._backend.run_and_wait(meas_calibs, shots=self._shots)
            self._meas_fitter[now] = CompleteMeasFitter(
                cal_results, state_labels, **self._cal_options
            )
        elif self._mitigation == "mthree":
            try:
                from mthree import M3Mitigation
            except ImportError as ex:
                raise MissingOptionalLibraryError(
                    libname="mthree",
                    name="execute",
                    pip_install="pip install mthree",
                ) from ex
            mit = M3Mitigation(self._backend.backend)
            mit.cals_from_system(shots=self._shots, **self._cal_options)
            self._meas_fitter[now] = mit
        self._time_threshold = now + self._refresh

    def _apply_mitigation(self, result: Result) -> list[Counts]:
        result_dt = self._datetime(result.date)
        fitters = [
            (abs(date - result_dt), date, fitter) for date, fitter in self._meas_fitter.items()
        ]
        _, min_date, min_fitter = min(fitters, key=lambda e: e[0])
        logger.info("apply mitigation data at %s", min_date)
        if self._mitigation in ["complete", "tensored"]:
            return min_fitter.filter.apply(result).get_counts()
        else:
            counts = result.get_counts()
            quasis = min_fitter.apply_correction(counts, self._cal_options["qubits"])
            ret = []
            for quasi, shots in zip(quasis, quasis.shots):
                ret.append(Counts({key: val * shots for key, val in quasi.items()}))
            return ret

    def apply_mitigation(self, results: list[Result]) -> list[list[Counts]]:
        """
        TODO
        """
        return [self._apply_mitigation(result) for result in results]

    def run_and_wait(
        self, circuits: Union[QuantumCircuit, list[QuantumCircuit]], **options
    ) -> Result:
        """
        TODO
        """
        self._maybe_calibrate()
        result = self._backend.run_and_wait(circuits, **options)
        self._maybe_calibrate()
        return result

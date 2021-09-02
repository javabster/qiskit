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

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from time import time
from typing import List, Union

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.ignis.mitigation.measurement import (
    CompleteMeasFitter,
    TensoredMeasFitter,
    complete_meas_cal,
    tensored_meas_cal,
)
from qiskit.providers.backend import BackendV1
from qiskit.result import Result

logger = logging.getLogger(__name__)


class BaseBackendWrapper(ABC):
    """
    TODO
    """
    @abstractmethod
    def run_and_wait(
        self, circuits: Union[QuantumCircuit, List[QuantumCircuit]], **options
    ) -> Result:
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


class BackendWrapper(BaseBackendWrapper):
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
        self, circuits: Union[QuantumCircuit, List[QuantumCircuit]], **options
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
    def to_backend(backend: Union[BackendV1, BaseBackendWrapper]) -> BackendV1:
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
        self, circuits: Union[QuantumCircuit, List[QuantumCircuit]], **options
    ) -> Result:
        """
        TODO
        """
        try:
            from qiskit.providers.ibmq.job import (
                IBMQJobFailureError,
                IBMQJobInvalidStateError,
            )
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
        self._refresh = refresh
        self._shots = shots
        self._time_threshold = 0.0
        self._cal_options = cal_options
        self._meas_fitter: dict[datetime, Union[CompleteMeasFitter, TensoredMeasFitter]] = {}

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
        now = time()
        if now <= self._time_threshold:
            return
        if self._mitigation == "tensored":
            meas_calibs, state_labels = tensored_meas_cal(**self._cal_options)
        elif self._mitigation == "complete":
            meas_calibs, state_labels = complete_meas_cal(**self._cal_options)

        logger.info("readout error mitigation calibration %s at %f", self._mitigation, now)
        cal_results = self._backend.run_and_wait(meas_calibs, shots=self._shots)

        dt = self._datetime(cal_results.date)
        if self._mitigation == "tensored":
            self._meas_fitter[dt] = TensoredMeasFitter(cal_results, **self._cal_options)
        elif self._mitigation == "complete":
            self._meas_fitter[dt] = CompleteMeasFitter(
                cal_results, state_labels, **self._cal_options
            )
        self._time_threshold = now + self._refresh

    def _apply_mitigation(self, result: Result):
        result_dt = self._datetime(result.date)
        fitters = [
            (abs(date - result_dt), date, fitter) for date, fitter in self._meas_fitter.items()
        ]
        _, min_date, min_fitter = min(fitters, key=lambda e: e[0])
        logger.info("apply mitigation data at %s", min_date)
        return min_fitter.filter.apply(result)

    def apply_mitigation(self, results: List[Result]):
        """
        TODO
        """
        return [self._apply_mitigation(result) for result in results]

    def run_and_wait(
        self, circuits: Union[QuantumCircuit, List[QuantumCircuit]], **options
    ) -> Result:
        """
        TODO
        """
        self._maybe_calibrate()
        result = self._backend.run_and_wait(circuits, **options)
        self._maybe_calibrate()
        return result

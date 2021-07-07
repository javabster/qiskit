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
Base Experiment Result class.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, List

from qiskit.providers import BaseJob
from qiskit.providers import JobV1
from qiskit.result import Result, Counts
from qiskit.exceptions import QiskitError

logger = logging.getLogger(__name__)


class Analysis(ABC):
    """Base experiment result analysis class."""
    def __init__(self,
                 experiment: str,
                 data: Optional[any] = None,
                 metadata: Optional[Dict[str, any]] = None):
        """Initialize the fitter"""
        # Experiment
        self._experiment = experiment
        self._unprocessed_data = False

        # Initialize with data
        self._exp_data = []
        self._exp_metadata = []
        self.add_data(data, metadata)

        # Cache for storing multiple calls to fitter on dataset
        self._cache_results = True

        # Store list of outputs of the `analyze` method
        self._results = []

        # Store list of size of data used when `analyze` was called, so
        # the subset of data can be recovered using `self.data[:size]`
        self._results_data_size = []

    @abstractmethod
    def _analyze(self,
                 data: List[any],
                 metadata: List[Dict[str, any]],
                 **params) -> any:
        """Run analysis of data"""
        # This should run the fitter and return the result
        # If the is no metadta, return None for metadata

    def _filter_data(self, data: Result, index: int) -> Counts:
        """Filter the required data from a Result.data dict"""
        # Derived classes should override this method to filter
        # only the required data.
        # The default behavior filters on counts.
        return data.get_counts(index)

    def __len__(self):
        """Return the number of stored results"""
        return len(self._results)

    @property
    def data(self):
        """Return stored data"""
        return self._exp_data

    @property
    def metadata(self):
        """Return stored metadata"""
        return self._exp_metadata

    @property
    def last_result(self):
        """Return most recent analysis result"""
        # Run analyze and return result
        if self._results:
            if self._unprocessed_data:
                logger.warning(
                    'ExperimentResult contains unprocessed data. Use `analyze` '
                    'to re-run analysis with all data.')
            return self._results[-1]
        raise QiskitError("No analysis results are stored. Use `analyze` method.")

    def add_data(self,
                 data: Union[BaseJob, Result, any],
                 metadata: Optional[Dict[str, any]] = None):
        """Add additional data to the fitter.

        Args:
                data: input data for the fitter.
                metadata: Optional, list of metadata dicts for input data.
                          if None will be taken from data Result object.
        """
        if isinstance(data, (BaseJob, JobV1)):
            data = data.result()

        if isinstance(data, Result):
            # Extract metadata from result object if not provided
            if metadata is None:
                if not hasattr(data.header, 'metadata'):
                    raise QiskitError("Experiment is missing metadata.")
                metadata = data.header.metadata
            # Get data from result
            new_data = []
            new_meta = []
            for i, meta in enumerate(metadata):
                if meta.get('experiment') == self._experiment:
                    new_data.append(self._filter_data(data, i))
                    new_meta.append(meta)
        else:
            # Add general preformatted data
            if not isinstance(data, list):
                new_data = [data]
            else:
                new_data = data

            if metadata is None:
                # Empty metadata incase it is not needed for a given experiment
                new_meta = len(new_data) * [{}]
            elif not isinstance(metadata, list):
                new_meta = [metadata]
            else:
                new_meta = metadata

        # Add extra data
        self._exp_data += new_data
        self._exp_metadata += new_meta
        self._unprocessed_data = True

        # Check metadata and data are same length
        if len(self._exp_metadata) != len(self._exp_data):
            raise QiskitError("data and metadata lists must be the same length")

    def analyze(self, **params):
        """Analyze the stored data.

        Returns:
            any: the output of the analysis,
        """
        # Run analysis with unprocessed adata
        result = self._analyze(self._exp_data, self._exp_metadata, **params)
        self._unprocessed_data = False
        if self._cache_results:
            self._results.append(result)
            self._results_data_size.append(len(self._exp_data))
        else:
            self._results = [result]
            self._results_data_size = [len(self._exp_data)]
        return result

    def result_data(self, i):
        """Return the list of data use for fitting a result"""
        return self._exp_data[:self._results_data_size[i]]

    def result_metadata(self, i):
        """Return the list of metadata use for fitting a result"""
        return self._exp_metadata[:self._results_data_size[i]]

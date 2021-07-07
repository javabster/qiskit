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
Expectation Value Experiment.
"""

from typing import Optional, Dict, Union, List, Callable

from qiskit import QuantumCircuit
from qiskit.result import Counts, Result
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Statevector

from .experiment import Experiment
from .analysis import Analysis
from .pauli_method import pauli_generator, pauli_analyzer
from .snapshot_method import snapshot_generator, snapshot_analyzer


class ExpectationValue(Experiment):
    """Expectation value experiment."""

    # pylint: disable=arguments-differ
    def __init__(self,
                 observable: BaseOperator,
                 method: Union[str, Callable] = 'Pauli',
                 initial_state: Optional[Union[QuantumCircuit, Statevector]] = None,
                 qubits: Optional[List[int]] = None):
        """Initialize expectation value experiment.

        Args:
            observable: an operator object for obserable.
            method: the measurement circuit conversion method for the observable.
                    See additional information.
            initial_state: Optional, the initial state quantum circuit. If
                           a Statevector or array is passed in it will be
                           converted to a circuit using the `initialize`
                           instruction. Can be set after initialization.
            qubits: Optional, the qubits to contract the operator on if the
                    initial state circuit is larger than the operator.

        Additional Information:
            Custom operator decomposition methods can be used by passing in a
            callable for the method. The signature of this callable should be:
            ``method(observable)`` where ``observable`` is an operator object.

            The default method ``"Pauli"`` will convert the input operator
            into a ``SparsePauliOp`` and perform a Pauli basis measurement
            for each Pauli component of the operator.
        """
        # Attributes
        self._op = observable
        self._meas_circuits = []
        self._metadata = []

        # Method selection
        if not isinstance(method, str):
            generator = method
        elif method == 'Pauli':
            generator = pauli_generator
        elif method == 'snapshot':
            generator = snapshot_generator
        else:
            raise QiskitError("Unrecognized ExpectationValue method: {}".format(method))

        # Generate circuits
        self._meas_circuits, metadata = generator(observable)
        super().__init__(self._meas_circuits[0].num_qubits)

        # Add metadata to base metadata
        base_meta = {'experiment': 'expval',
                     'qubits': None,
                     'method': str(method)}
        for meta in metadata:
            new_meta = base_meta.copy()
            for key, val in meta.items():
                new_meta[key] = val
            self._metadata.append(new_meta)

        # Set optional initial circuit
        # This can also be set later after initialization
        self._qubits = None
        self._initial_circuit = None
        if initial_state is not None:
            self.set_initial_state(initial_state, qubits=qubits)

    def set_initial_state(self,
                          initial_state: Union[QuantumCircuit, Statevector],
                          qubits: Optional[List[int]] = None):
        """Set initial state for the expectation value.

        Args:
            initial_state: Optional, the initial state quantum circuit. If
                           a Statevector or array is passed in it will be
                           converted to a circuit using the `initialize`
                           instruction. Can be set after initialization.
            qubits: Optional, the qubits to contract the operator on if the
                    initial state circuit is larger than the operator.

        Raises:
            QiskitError: if the initial state is invalid.
            QiskitError: if the number of qubits does not match the observable.
        """
        if isinstance(initial_state, QuantumCircuit):
            self._initial_circuit = initial_state
        else:
            initial_state = Statevector(initial_state)
            num_qubits = initial_state.num_qubits
            self._initial_circuit = QuantumCircuit(num_qubits)
            self._initial_circuit.initialize(initial_state.data, list(range(num_qubits)))

        self._num_qubits = self._initial_circuit.num_qubits
        self.set_qubits(qubits)

    def set_qubits(self, qubits: List[int]):
        """Set qubits to contract the operator on.

        Args:
            qubits: Optional, the qubits to contract the operator on if the
                    initial state circuit is larger than the operator.

        Raises:
            QiskitError: if the number of qubits does not match the observable."""
        if qubits is not None and len(qubits) != self._op.num_qubits:
            raise QiskitError('Number of qubits does not match operator '
                              '{} != {}'.format(len(qubits), self._op.num_qubits))
        self._qubits = qubits
        for meta in self._metadata:
            meta['qubits'] = qubits

    def circuits(self) -> List[QuantumCircuit]:
        """Generate a list of experiment circuits."""
        expval_circuits = []
        num_qubits = self._initial_circuit.num_qubits
        for meas_circ in self._meas_circuits:
            num_clbits = meas_circ.num_qubits
            circ = QuantumCircuit(num_qubits, num_clbits)
            circ.compose(self._initial_circuit, inplace=True)
            circ.compose(meas_circ, qubits=self._qubits, inplace=True)
            expval_circuits.append(circ)
        return expval_circuits

    def metadata(self) -> List[dict]:
        """Generate a list of experiment circuits metadata."""
        return self._metadata


class ExpectationValueAnalysis(Analysis):
    """Expectation value experiment analysis."""

    # pylint: disable=arguments-differ
    def __init__(self,
                 data: Optional[any] = None,
                 metadata: Optional[Dict[str, any]] = None,
                 method: Optional[Union[str, Callable]] = 'Pauli',
                 mitigator: Optional = None):
        """Initialize expectation value experiment.

        Args:
            data: Optional, result data to initialize with.
            metadata: Optional, result metadata to initialize with.
            method: Optional, the analysis method. See additional information.
            mitigator: Optional, measurement error mitigator object to apply
                       mitigation.

        Additional Information:
            Custom analysis methods can be supplied using a callable for the
            ``method`` kwarg. The signature of this callable should be:
            ``method(data, metadata, mitigator)`` where ``data`` is a list of
            :class:`Counts` objects, ``metadata`` is a list of metadata dicts,
            and ``mitigator`` is either a measurement error mitigator object,
            or ``None`` for no mitigation.

            The default method ``"Pauli"`` assumes all counts correspond to
            Pauli basis measurements. The measurement basis and coefficient
            should be stored in the metadata under the fields ``"basis"`` and
            ``"coeff"`` respectively.

            * If the basis field is not present a default basis of measuring in
              the Z-basis on all qubits is used.
            * If the coefficient field is not present it is assumed to be 1.
        """
        # Measurement Error Mitigation
        self._mitigator = mitigator

        # Set analyze function for method
        if not isinstance(method, str):
            self._method = method
        elif method == 'Pauli':
            self._method = pauli_analyzer
        elif method == 'snapshot':
            self._method = snapshot_analyzer
        else:
            raise QiskitError("Unrecognized ExpectationValue method: {}".format(method))

        # Base Experiment Result class
        super().__init__('expval', data=data, metadata=metadata)

    def _analyze(self,
                 data: List[Counts],
                 metadata: List[Dict[str, any]],
                 mitigator: Optional = None):
        """Fit and return the Mitigator object from the calibration data."""
        if mitigator is None:
            mitigator = self._mitigator
        return self._method(self._exp_data, self._exp_metadata, mitigator=mitigator)

    def _filter_data(self, data: Result, index: int) -> Counts:
        """Filter the required data from a Result.data dict"""
        if self._method == snapshot_analyzer:
            # For snapshots we don't use counts
            snapshots = data.data(index).get('snapshots', {}).get('expectation_value', {})
            snapshots['shots'] = data.results[index].shots
            return snapshots
        # Otherwise we return counts
        return super()._filter_data(data, index)

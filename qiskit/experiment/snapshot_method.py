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
Snapshot Expectation Value Experiment.
"""

import logging
from typing import Optional, Dict, Union, List, Tuple

import numpy as np

from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import SparsePauliOp, Operator

logger = logging.getLogger(__name__)


def snapshot_generator(observable: Union[SparsePauliOp, Operator]) -> Tuple[
        List[QuantumCircuit], List[Dict[str, any]]]:
    """Circuit generator for expectation value snapshots."""
    if isinstance(observable, SparsePauliOp):
        pauli_op = observable
    else:
        pauli_op = SparsePauliOp.from_operator(observable)

    # Get snapshot params for operator
    params = [[coeff, pauli] for pauli, coeff in pauli_op.label_iter()]

    # Get snapshot params for operator ** 2
    pauli_op_sq = pauli_op.dot(pauli_op).simplify()
    params_sq = [[coeff, pauli] for pauli, coeff in pauli_op_sq.label_iter()]

    num_qubits = pauli_op.num_qubits
    qubits = list(range(num_qubits))

    circuit = QuantumCircuit(num_qubits)
    circuit.snapshot('expval', snapshot_type='expectation_value_pauli',
                     qubits=qubits, params=params)
    circuit.snapshot('sq_expval', snapshot_type='expectation_value_pauli',
                     qubits=qubits, params=params_sq)

    return [circuit], [{}]


def snapshot_analyzer(data: List[Dict],
                      metadata: List[Dict[str, any]],
                      mitigator: Optional = None):
    """Fit expectation value from snapshots."""
    if mitigator is not None:
        logger.warning('Error mitigation cannot be used with the snapshot'
                       ' expectation value method.')

    if len(data) != 1:
        raise QiskitError("Invalid list of data")

    snapshots = data[0]
    meta = metadata[0]

    if 'expval' not in snapshots or 'sq_expval' not in snapshots:
        raise QiskitError("Snapshot keys missing from snapshot dict.")

    expval = snapshots['expval'][0]['value']
    sq_expval = snapshots['sq_expval'][0]['value']

    # Convert to real if imaginary part is zero
    if np.isclose(expval.imag, 0):
        expval = expval.real
    if np.isclose(sq_expval.imag, 0):
        sq_expval = sq_expval.real

    # Compute variance and standard error
    variance = sq_expval - expval ** 2

    # Get shots
    if 'shots' in meta:
        shots = meta['shots']
    else:
        shots = snapshots.get('shots', 1)

    stderror = np.sqrt(variance / shots)

    return expval, stderror

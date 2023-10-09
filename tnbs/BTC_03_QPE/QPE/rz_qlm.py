"""

Author: Gonzal Ferro
"""

import time
import qat.lang.AQASM as qlm
from qpe import CQPE

def qpe_rz_qlm(angles, auxiliar_qbits_number, shots=0, qpu=None):
    """
    Computes the Quantum Phase Estimation for a Rz Kroneckr product

    Parameters
    __________

    angles : list
        list with the angles that are applied to each qubit of the circuit
    auxiliar_qbits_number : int
        number of auxiliar qubits for doing QPE
    shots : int
        number of shots for gettiong the results. 0 for exact solution
    qpu : Atos QLM QPU object
        QLM QPU for solving the circuit

    Returns
    _______

    results : pandas DataFrame
        pandas DataFrame with the distribution of the eigenvalues with
        a bin discretization of 2^auxiliar_qbits_number
        * lambda : bin discretization for eigenvalues based on the
        discretization input (auxiliar_qbits_number input)
        * Probability: probability of finding any eigenvalue inside
        of the correspoondent lambda bin

    qft_pe : CQPE object

    """
    n_qbits = len(angles)
    #print('n_qubits: {}'.format(n_qbits))
    initial_state = qlm.QRoutine()
    q_bits = initial_state.new_wires(n_qbits)

    # Creating the superposition initial state
    for i in range(n_qbits):
        #print(i)
        initial_state.apply(qlm.H, q_bits[i])

    # Creating the Operator Rz_n
    rzn_gate = rz_angles(angles)
    #We create a python dictionary for configuration of class
    qft_pe_dict = {
        'initial_state': initial_state,
        'unitary_operator': rzn_gate,
        'qpu' : qpu,
        'auxiliar_qbits_number' : auxiliar_qbits_number,
        'complete': True,
        'shots' : shots
    }
    qft_pe = CQPE(**qft_pe_dict)
    qft_pe.run()
    qft_pe_results = qft_pe.result
    qft_pe_results.sort_values('lambda', inplace=True)
    results = qft_pe_results[['lambda', 'Probability']]
    return results, qft_pe

def rz_angles(thetas):
    """
    Creates a QLM abstract Gate with a R_z^n operator of an input array of angles

    Parameters
    __________

    thetas : array
        Array with the angles of the R_z^n operator

    Returns
    _______

    r_z_n : QLM AbstractGate
        AbstractGate with the implementation of R_z_^n of the input angles

    """
    n_qbits = len(thetas)

    @qlm.build_gate("Rz_{}".format(n_qbits), [], arity=n_qbits)
    def rz_routine():
        routine = qlm.QRoutine()
        q_bits = routine.new_wires(n_qbits)
        for i in range(n_qbits):
            routine.apply(qlm.RZ(thetas[i]), q_bits[i])
        return routine
    r_z_n = rz_routine()
    return r_z_n


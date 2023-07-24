"""
All mandatory functionsfor executing theoretical and atos qlm simulation
for computing eigenvalues of a R_z^n operator

Author: Gonzal Ferro
"""

import numpy as np
import pandas as pd
import qat.lang.AQASM as qlm
from QPE.qpe import CQPE


def get_qpu(qpu=None):
    """
    Function for selecting solver.

    Parameters
    ----------

    qpu : str
        * qlmass: for trying to use QLM as a Service connection to CESGA QLM
        * python: for using PyLinalg simulator.
        * c: for using CLinalg simulator

    Returns
    ----------

    linal_qpu : solver for quantum jobs
    """

    if qpu is None:
        raise ValueError(
            "qpu CAN NOT BE NONE. Please select one of the three" +
            " following options: qlmass, python, c")
    if qpu == "qlmass":
        try:
            from qlmaas.qpus import LinAlg
            linalg_qpu = LinAlg()
        except (ImportError, OSError) as exception:
            raise ImportError(
                "Problem Using QLMaaS. Please create config file" +
                "or use mylm solver") from exception
    elif qpu == "python":
        from qat.qpus import PyLinalg
        linalg_qpu = PyLinalg()
    elif qpu == "c":
        from qat.qpus import CLinalg
        linalg_qpu = CLinalg()
    elif qpu == "default":
        from qat.qpus import get_default_qpu
        linalg_qpu = get_default_qpu()
    else:
        raise ValueError(
            "Invalid value for qpu. Please select one of the three "+
            "following options: qlmass, python, c")
    #print("Following qpu will be used: {}".format(linalg_qpu))
    return linalg_qpu

# Functions for generating theoretical eigenvalues of R_z^n
def bitfield(n_int: int, size: int):
    """Transforms an int n_int to the corresponding bitfield of size size

    Parameters
    ----------
    n_int : int
        integer from which we want to obtain the bitfield
    size : int
        size of the bitfield

    Returns
    ----------
    full : list of ints
        bitfield representation of n_int with size size

    """
    aux = [1 if digit == "1" else 0 for digit in bin(n_int)[2:]]
    right = np.array(aux)
    left = np.zeros(max(size - right.size, 0))
    full = np.concatenate((left, right))
    return full.astype(int)

def rz_eigenv_from_state(state, angles):
    """
    For a fixed input state and the angles of the R_z^n operator compute
    the correspondent eigenvalue.

    Parameters
    __________

    state : np.array
        Array with the binnary representation of the input state
    angles: np.array
        Array with the angles for the R_z^n operator.

    Returns
    _______

    lambda_ : float
        The eigenvalue for the input state of the R_z^n operator with
        the input angles

    """
    new_state = np.where(state == 1, -1, 1)
    # Computing the eigenvalue correspondent to the input state
    thetas = - 0.5 * np.dot(new_state, angles)
    # We want the angle between [0, 2pi]
    thetas_2pi = np.mod(thetas, 2 * np.pi)
    # Normalization of the angle between [0,1]
    lambda_ = thetas_2pi / (2.0 * np.pi)
    return lambda_

def rz_eigv(angles):
    """
    Computes the complete list of eigenvalues for a R_z^n operator
    for ainput list of angles
    Provides the histogram between [0,1] with a bin of 2**discretization
    for the distribution of eigenvalues of a R_z^n operator for a given
    list of angles.

    Parameters
    __________

    angles: np.array
        Array with the angles for the R_z^n operator.

    Returns
    _______

    pdf : pandas DataFrame
        DataFrame with all the eigenvalues of the R_z^n operator for
        the input list angles. Columns
            * States : Eigenstates of the Rz^n operator (least
                significative bit is leftmost)
            * Int_lsb_left : Integer conversion of the state
                (leftmost lsb)
            * Int_lsb_rightt : Integer conversion of the state
                (rightmost lsb)
            * Eigenvalues : correspondent eigenvalue

    """

    n_qubits = len(angles)
    # Compute eigenvalues of all posible eigenstates
    eigv = [rz_eigenv_from_state(bitfield(i, n_qubits), angles)\
        for i in range(2**n_qubits)]
    pdf = pd.DataFrame(
        [eigv],
        index=['Eigenvalues']
    ).T
    pdf['Int_lsb_left'] = pdf.index
    state = pdf['Int_lsb_left'].apply(
        lambda x: bin(x)[2:].zfill(n_qubits)
    )
    pdf['States'] = state.apply(lambda x: '|' + x[::-1] + '>')
    pdf['Int_lsb_right'] = state.apply(lambda x: int('0b'+x[::-1], base=0))
    pdf = pdf[['States', 'Int_lsb_left', 'Int_lsb_right', 'Eigenvalues']]
    return pdf

def make_histogram(eigenvalues, discretization):
    """
    Given an input list of eigenvalues compute the correspondent
    histogram using a bins = 2^discretization

    Parameters
    __________

    eigenvalues : list
        List with the eigenvalues
    discretization: int
        Histogram discretization parameter: The number fo bins for the
        histogram will be: 2^discretization

    Returns
    _______

    pdf : pandas DataFrame
        Pandas Dataframe with the 2^m bin frequency histogram for the
        input list of eigenvalues. Columns
            * lambda : bin discretization for eigenvalues based on the
                discretization input
            * Probability: probability of finding any eigenvalue inside
                of the correspoondent lambda bin
    """

    # When the histogram is computed can be some problems with numeric
    # approaches. So we compute the maximum number of decimals for
    # a bare discretization of the bins and use it for rounding properly
    # the eigenvalues
    lambda_strings = [len(str(i / 2 ** discretization).split('.')[1]) \
        for i in range(2 ** discretization)]
    decimal_truncation = max(lambda_strings)
    trunc_eigenv = [round(i_, decimal_truncation) for i_ in list(eigenvalues)]
    pdf = pd.DataFrame(
        np.histogram(
            trunc_eigenv,
            bins=2 ** discretization,
            range=(0, 1.0)
        ),
        index=["Counts", "lambda"]
    ).T[:2 ** discretization]
    pdf["Probability"] = pdf["Counts"] / sum(pdf["Counts"])
    pdf.drop(columns=['Counts'], inplace=True)

    return pdf

# Below are functions forAtos myqlm simulation of R_z^n
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
    results.reset_index(drop=True, inplace=True)
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

def computing_shots(pdf):
    """
    Compute the number of shots. The main idea is that the samples for
    the lowest degeneracy eigenvalues will be enough. In this case
    enough is that that we measured an eigenvalue that will have an
    error from respect to the theorical one lower than the
    discretization precision at least 100 times

    Parameters
    __________

    pdf : pandas DataFrame
        DataFrame with the theoretical eigenvalues

    Returns
    _______

    shots : int
        number of shots for QPE algorithm

    """
    # prob of less frequent eigenvalue
    lfe = min(pdf.value_counts('Eigenvalues')) / len(pdf)
    shots = int((1000 / (lfe * 0.81))) + 1
    return shots

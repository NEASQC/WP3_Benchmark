"""
This module contains all the functions in order to load data into the
quantum state.
There are two implementations for the loading of a function:

    * one based on brute force
    * one based on multiplexors.

The implementation of the multiplexors is a non-recursive version of:

    V.V. Shende, S.S. Bullock, and I.L. Markov.
    Synthesis of quantum-logic circuits.
    IEEE Transactions on Computer-Aided Design of Integrated Circuits
    and Systems, 25(6):1000–1010, Jun 2006
    arXiv:quant-ph/0406176v5

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro

"""

import sys
import time
import random
import numpy as np
import qat.lang.AQASM as qlm
from qat.lang.models import KPTree
from scipy.stats import norm
sys.path.append('../')
from PL.utils.utils import bitfield, left_conditional_probability, fwht
from PL.utils.data_extracting import get_results


def mask(number_qubits, index):
    r"""
    Transforms the state :math:`|index\rangle` into the state
    :math:`|11...1\rangle` of size number qubits.

    Parameters
    ----------
    number_qubits : int
    index : int

    Returns
    ----------
    mask : Qlm abstract gate
        the gate that we have to apply in order to transform
        state :math:`|index\rangle`. Note that it affects all states.
    """
    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits)
    bits = bitfield(index, number_qubits)
    for k in range(number_qubits):
        if bits[-k - 1] == 0:
            routine.apply(qlm.X, quantum_register[k])

    return routine

def load_probability(
    probability_array: np.array,
    method: str = "multiplexor",
    id_name: str = str(time.time_ns())
):
    """
    Creates a QLM Abstract gate for loading a given discretized probability
    distribution using Quantum Multiplexors.

    Parameters
    ----------
    probability_array : numpy array
        Numpy array with the discretized probability to load. The arity of
        of the gate is int(np.log2(len(probability_array))).
    method : str
        type of loading method used:
            multiplexor : with quantum Multiplexors
            brute_force : using multicontrolled rotations by state
    id_name : str
        name for the Abstract Gate

    Returns
    ----------

    P_Gate :  AbstractGate
        Customized Abstract Gate for Loading Probability array using
        Quantum Multiplexors
    """
    number_qubits = int(np.log2(probability_array.size))

    @qlm.build_gate("P_{" + id_name + "}", [], arity=number_qubits)
    def load_probability_gate():
        """
        QLM Routine generation.
        """
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)
        # Now go iteratively trough each qubit computing the
        # probabilities and adding the corresponding multiplexor
        for m_qbit in range(number_qubits):
            # print(m)
            # Calculates Conditional Probability
            conditional_probability = left_conditional_probability(
                m_qbit, probability_array
            )
            # Rotation angles: length: 2^(i-1)-1 and i the number of
            # qbits of the step
            thetas = 2.0 * (np.arccos(np.sqrt(conditional_probability)))
            if m_qbit == 0:
                # In the first iteration it is only needed a RY gate
                routine.apply(qlm.RY(thetas[0]), register[number_qubits - 1])
            else:
                # In the following iterations we have to apply
                # multiplexors controlled by m_qbit qubits
                # We call a function to construct the multiplexor,
                # whose action is a block diagonal matrix of Ry gates
                # with angles theta
                routine.apply(
                    #multiplexor_ry(thetas),
                    load_angles(thetas, method),
                    register[number_qubits - m_qbit : number_qubits],
                    register[number_qubits - m_qbit - 1],
                )
        return routine

    return load_probability_gate()

def load_angles(angles: np.array, method: str = "multiplexor"):
    r"""
    This function serves as an interface for the two different implementations
    of multi controlled rotations: load_angles_brute_force and multiplexor_RY.

    Notes
    -----
    .. math::
        |\Psi\rangle = \sum_{j=0}^{2^n-1}\alpha_j|j\rangle\otimes|0\rangle

    .. math::
        \mathcal{load\_angles}([\theta_j]_{j=0,1,2...2^n-1})|\Psi\rangle \
        =\sum_{j=0}^{2^n-1}\alpha_j|j\rangle\otimes \
        \big(\cos(\theta_j)|0\rangle+\sin(\theta_j)|1\rangle\big)

    Parameters
    ----------
    angles : numpy array
        Angles to load in the circuit. The arity of the gate is:
        int(np.log2(len(angle)))+1.
    method : string
        Method used in the loading. Default method.
    """
    number_qubits = int(np.log2(angles.size)) + 1
    if np.max(angles) > 2 * np.pi:
        raise ValueError("ERROR: function f not properly normalised")
    if angles.size != 2 ** (number_qubits - 1):
        print("ERROR: size of function f is not a factor of 2")
    if method == "brute_force":
        routine = load_angles_brute_force(angles)
    else:
        routine = multiplexor_ry(angles)
    return routine

def load_angles_brute_force(angles: np.array):
    r"""
    Given a list of angles this function creates a QLM routine that applies
    rotations of each angle of the list, over an auxiliary qubit, controlled
    by the different states of the measurement basis.
    Direct QLM multi controlled rotations were used for the implementation.

    Notes
    -----
    .. math::
        |\Psi\rangle = \sum_{j=0}^{2^n-1}\alpha_j|j\rangle\otimes|0\rangle
    .. math::
        \mathcal{load\_angles\_brute\_force} \
        ([\theta_j]_{j=0,1,2...2^n-1}) |\Psi\rangle=\sum_{j=0}^{2^n-1} \
        \alpha_j|j\rangle\otimes\big(\cos(\theta_j)|0\rangle+ \
        \sin(\theta_j)|1\rangle\big)


    Parameters
    ----------
    angles : numpy array
        Angles to load in the circuit. The arity of the gate is:
        int(np.log2(len(angle)))+1.
    """
    number_qubits = int(np.log2(angles.size)) + 1
    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits)
    for i in range(angles.size):
        routine.apply(load_angle(number_qubits, i, angles[i]), quantum_register)
    return routine

def multiplexor_ry(angles: np.array, ordering: str = "sequency"):
    r"""
    Given a list of angles this functions creates a QLM routine that applies
    rotations of each angle of the list, over an auxiliary qubit, controlled
    by the different states of the measurement basis.
    The multi-controlled rotations were implemented using Quantum Multiplexors.

    Notes
    -----
    .. math::
        |\Psi\rangle = \sum_{j=0}^{2^n-1}\alpha_j|j\rangle\otimes|0\rangle
    .. math::
        \mathcal{multiplexor\_RY} \
        ([\theta_j]_{j=0,1,2...2^n-1})|\Psi\rangle = \sum_{j=0}^{2^n-1} \
        \alpha_j|j\rangle\otimes\big(\cos(\theta_j)|0\rangle+\sin(\theta_j)|1\rangle\big)

    Parameters
    ----------
    angles : numpy array
        Angles to load in the circuit. The arity of the gate is:
            int(np.log2(len(angle)))+1.
    """
    number_qubits = int(np.log2(angles.size))
    angles = fwht(angles, ordering=ordering)
    angles = angles / 2**number_qubits
    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits + 1)
    control = np.zeros(2**number_qubits, dtype=int)
    for i in range(number_qubits):
        for j in range(2**i - 1, 2**number_qubits, 2**i):
            control[j] = number_qubits - i - 1
    for i in range(2**number_qubits):
        routine.apply(qlm.RY(angles[i]), quantum_register[number_qubits])
        routine.apply(
            qlm.CNOT, quantum_register[control[i]], quantum_register[number_qubits]
        )
    return routine

@qlm.build_gate("LP", [int, int, float], arity=lambda x, y, z: x)
def load_angle(number_qubits: int, index: int, angle: float):
    r"""
    Creates an QLM Abstract Gate that apply a rotation of a given angle
    into a auxiliary qubit controlled by a given state of the measurement basis.
    Direct QLM multi controlled rotations were used for the implementation.

    Notes
    -----
    .. math::
        |\Psi\rangle = \sum_{j=0}^{2^n-1}\alpha_j|j\rangle\otimes|0\rangle
    .. math::
        \mathcal{load\_angle}(\theta, |i\rangle)|\Psi\rangle \
        =\sum_{j=0, j\ne i}^{2^n-1}\alpha_j|j\rangle\otimes|0\rangle+ \
        \alpha_i|i\rangle\otimes\big(\cos(\theta)|0\rangle+\sin(\theta) \
        |1\rangle\big)


    Parameters
    ----------
    number_qubits : int
        Number of qubits for the control register. The arity of the gate is number_qubits+1.
    index : int
        Index of the state that we control.
    angle : float
        Angle that we load.
    """

    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits)

    routine.apply(mask(number_qubits - 1, index), quantum_register[: number_qubits - 1])
    routine.apply(
        qlm.RY(angle).ctrl(number_qubits - 1),
        quantum_register[: number_qubits - 1],
        quantum_register[number_qubits - 1],
    )
    routine.apply(mask(number_qubits - 1, index), quantum_register[: number_qubits - 1])

    return routine

def get_theoric_probability(n_qbits: int) -> (np.ndarray, np.ndarray, float, float, float, int):
    """
    Get the discretization of the PDF for N qubits
    """
    mean = random.uniform(-2., 2.)
    sigma = random.uniform(0.1, 2.)

    intervals = 2 ** n_qbits

    ppf_min = 0.005
    ppf_max = 0.995
    norma = norm(loc=mean, scale=sigma)
    x_ = np.linspace(norma.ppf(ppf_min), norma.ppf(ppf_max), num=intervals)
    step = x_[1] - x_[0]

    data = norma.pdf(x_)
    data = data/np.sum(data)
    mindata = np.min(data)
    shots = min(1000000, max(10000, round(100/mindata)))
    #data = np.sqrt(data)
    return x_, data, mean, sigma, float(step), shots, norma

def get_qlm_probability(data, load_method, shots, qpu):
    """
    executing quantum stuff
    """
    if load_method == "multiplexor":
        p_gate = load_probability(data, method="multiplexor")
    elif load_method == "brute_force":
        p_gate = load_probability(data, method="brute_force")
    elif load_method == "KPTree":
        p_gate = KPTree(np.sqrt(data)).get_routine()
    else:
        error_text = "Not valid load_method argument."\
            "Select between: multiplexor, brute_force or KPTree"
        raise ValueError(error_text)
    tick = time.time()
    result, circuit, _, _ = get_results(
        p_gate,
        linalg_qpu=qpu,
        shots=shots
    )
    tack = time.time()
    quantum_time = tack - tick

    if load_method == "KPTree":
        #Use different order convention
        result.sort_values(by="Int", inplace=True)
    return result, circuit, quantum_time

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

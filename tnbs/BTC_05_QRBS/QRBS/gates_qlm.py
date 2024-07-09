"""
QLM quantum implementation of inferential Gates.
Based on the paper:

    Moret-Bonillo, Vicente and Magaz-Romero, Samuel and Mosqueira-Rey, Eduardo
    Quantum Computing for Dealing with Inaccurate Knowledge Related
    to the Certainty Factors Model
    Mathematics, volumen 10, 2022

"""
import numpy as np
import qat.lang.AQASM as qlm
from qat.lang.models import KPTree


def init_state(alpha):
    r"""
    Qubit state initialization.

    Notes
    -----
    .. math::
        |\Psi\rangle = \sqrt{1-\alpha} |0\rangle + \sqrt{\alpha} |1\rangle

    Parameters
    ----------
    alpha : float
       Probability of getting |1>

    Return
    ------
    gate : QLM gate
        QLM gate that encodes a quantum state such that the probability
        of getting the state |1> is alpha.
    """
    gate = KPTree(np.array([np.sqrt(1.0 - alpha), np.sqrt(alpha)])).get_routine()
    return gate


def _matrix_gen(inaccuracy):
    r"""
    Definition of the matrix for the M gate

    Notes
    -----
    .. math::
        \theta = \frac{\text{inaccuracy}\pi}{2}
    .. math::
        M = \begin{bmatrix}
            \sin{\theta} & \cos{\theta} \\
            \cos{\theta} & -\sin{\theta} \\
        \end{bmatrix}

    Parameters
    ----------
    inaccuracy : float
        Input inacuracy for the matrix
    Return
    ------
    gate : numpy array
        array with the Matrix for M gate
    """
    theta = inaccuracy * np.pi / 2
    return np.array([
        [np.cos(theta), np.sin(theta)],
        [np.sin(theta), -np.cos(theta)]
    ])

m_gate = qlm.AbstractGate("M", [float], arity=1, matrix_generator=_matrix_gen)

@qlm.build_gate("Precision", [float], arity=1)
def precision_gate(precision):
    """
    Quantum QLM gate for a precision fact.
    Parameters
    ----------
    precision : float
        Precision of the fact
    Return
    ------
    routine : QLM Routine
        QLM Routine with the quantum implementation for a fact with
        precision
    """
    routine = qlm.QRoutine()
    register = routine.new_wires(1)
    routine.apply(m_gate(precision), register[0])
    return routine

@qlm.build_gate("Rule", [float], arity=3)
def rule_gate(certainty):
    """
    Quantum QLM gate for a rule with a input level of certainty
    Parameters
    ----------
    certainty : float
        Certainty of the rule
    Return
    ------
    routine : QLM Routine
        QLM Routine with the quantum implementation for a rule with
        certainty
    """

    routine = qlm.QRoutine()
    register = routine.new_wires(3)
    routine.apply(m_gate(certainty), register[1])
    routine.apply(qlm.CCNOT, register[0], register[1], register[2])
    return routine

@qlm.build_gate("Not", [], arity=2)
def not_gate():
    """
    Quantum QLM gate for a NOT inferential gate
    Return
    ------
    routine : QLM Routine
        QLM Routine with the quantum implementation for a NOT gate
    """
    routine = qlm.QRoutine()
    register = routine.new_wires(2)
    routine.apply(qlm.CNOT, register[0], register[1])
    routine.apply(qlm.X, register[1])
    return routine

@qlm.build_gate("And", [], arity=3)
def and_gate():
    """
    Quantum QLM gate for a AND inferential gate
    Return
    ------
    routine : QLM Routine
        QLM Routine with the quantum implementation for a AND gate
    """
    routine = qlm.QRoutine()
    register = routine.new_wires(3)
    routine.apply(qlm.CCNOT, register[0], register[1], register[2])
    return routine

@qlm.build_gate("Or", [], arity=3)
def or_gate():
    """
    Quantum QLM gate for a OR inferential gate
    Return
    ------
    routine : QLM Routine
        QLM Routine with the quantum implementation for a OR gate
    """
    routine = qlm.QRoutine()
    register = routine.new_wires(3)
    routine.apply(qlm.CCNOT, register[0], register[1], register[2])
    routine.apply(qlm.CNOT, register[0], register[2])
    routine.apply(qlm.CNOT, register[1], register[2])
    return routine



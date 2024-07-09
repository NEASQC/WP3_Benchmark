"""
Python Classes for implementing Quantum QLM versions of the inferential
gates.
The base class is the InferentialGate one.


"""

from random import random
from scipy.optimize import minimize
import numpy as np
import qat.lang.AQASM as qlm
from gates_qlm import init_state, precision_gate, rule_gate
from gates_qlm import not_gate, and_gate, or_gate
from solve import solve_qjob



class InferentialGate:
    """
    Inferential gate base class
    Parameters
    ----------
    qpu : QLM qpu
        QLM qpu for solving the QLM quantum circuits
    alpha_c : float
        Desired probability of the |1> state for the gate
    """

    def __init__(self, qpu, alpha_c):

        # Coefficient of the |1> component of the |a> state
        self.alpha_a = None
        # Coefficient of the |1> component of the |b> state
        self.alpha_b = None
        # Coefficient of the |1> component of the |c> state
        self.alpha_c = alpha_c
        # Inputs
        self.qpu = qpu

        # Initial guess for minimization
        self.guess_ = None
        self.job = None
        self.loss_ = None
        self.gate = None

        self.factor = 0.2

    def fit(self):
        """
        Method for executing the minimization of the gate
        """
        self.guess()
        self.minimize()

    def guess(self):
        """
        Method for initializing guess variables for minimization
        """
        pass
    def circuit(self, alpha, beta):
        """
        Method for building the corresponding quantum circuit
        Parameters
        ----------
        alpha : float
            Depending on the gate it can be: Probability of |1> state
            for first input qubit or the precision of the fact for the
            gate. It can be None
            Probability of |1> state for first input qubit
        beta : float
            Depending on the gate it can be: Probability of |1> state
            for second input qubit or the certainty of the rule for the
            gate. It can be None
        """
        pass
    def loss(self, guess, output):
        """
        Method for computing the loss
        Parameters
        ----------
        guess : list
            list with the inputs variables for the method circuit.
        output : float
            Desired probability of |1> state for the gate.
        """
        pass

    def minimize(self):
        """
        Method for minimize the parameters of the QLM quantum gate
        """
        pass

class AND(InferentialGate):
    """
    AND gate class.
    Parameters
    ----------
    qpu : QLM qpu
        QLM qpu for solving the QLM quantum circuits
    alpha_c : float
        Desired probability of the |1> state for the AND gate
    """

    def guess(self):
        """
        Method for initializing guess variables for minimization
        """
        self.alpha_a = random()
        self.alpha_b = random()
        condition = min(self.alpha_a, self.alpha_b)
        lower_limit = (1.0 - self.factor) * self.alpha_c
        higher_limit = (1.0 + self.factor) * self.alpha_c
        while (condition < lower_limit) or (condition > higher_limit):
            self.alpha_a = random()
            self.alpha_b = random()
            condition = min(self.alpha_a, self.alpha_b)

        # Inital guess for optimization
        self.guess_ = np.array([self.alpha_a, self.alpha_b])
    def circuit(self, alpha, beta):
        """
        Method for building the corresponding quantum circuit for AND gate
        Parameters
        ----------
        alpha : float
            Probability of |1> state for first input qubit
        beta : float
            Probability of |1> state for second input qubit.
        """
        routine = qlm.QRoutine()
        gate = and_gate()
        register = routine.new_wires(gate.arity)
        routine.apply(init_state(alpha), register[0])
        routine.apply(init_state(beta), register[1])
        routine.apply(gate, register)
        self.job = routine.to_circ().to_job(
            nbshots=0,
            qubits=[gate.arity - 1]
        )
    def loss(self, guess, output):
        """
        Method for computing the loss of the AND gate
        Parameters
        ----------
        guess : list
            list with the inputs variables for the AND circuit.
        output : float
            Desired probability of |1> state for the AND gate.
        """
        alpha = guess[0]
        beta = guess[1]
        self.circuit(alpha, beta)
        pdf = solve_qjob(self.job, self.qpu)
        loss_ = (pdf[pdf["Int"] == 1]["Probability"] - output) ** 2
        return loss_
    def minimize(self):
        """
        Method for optimize the AND QLM quantum gate
        """
        bnds = ((0.0, 1.0) for i in self.guess_)
        res = minimize(
            self.loss,
            self.guess_,
            args=self.alpha_c,
            bounds=bnds
        )
        self.alpha_a = res.x[0]
        self.alpha_b = res.x[1]


class OR(InferentialGate):
    """
    OR gate class.
    Parameters
    ----------
    qpu : QLM qpu
        QLM qpu for solving the QLM quantum circuits
    alpha_c : float
        Desired probability of the |1> state for the OR gate
    """

    def guess(self):
        """
        Method for initializing guess variables for minimization
        """
        self.alpha_a = random()
        self.alpha_b = random()
        condition = max(self.alpha_a, self.alpha_b)
        lower_limit = (1.0 - self.factor) * self.alpha_c
        higher_limit = (1.0 + self.factor) * self.alpha_c
        while (condition < lower_limit) or (condition > higher_limit):
            self.alpha_a = random()
            self.alpha_b = random()
            condition = max(self.alpha_a, self.alpha_b)

        # Inital guess for optimization
        self.guess_ = np.array([self.alpha_a, self.alpha_b])


    def circuit(self, alpha, beta):
        """
        Method for building the corresponding quantum circuit for OR gate
        Parameters
        ----------
        alpha : float
            Probability of |1> state for first input qubit
        beta : float
            Probability of |1> state for second input qubit.
        """
        routine = qlm.QRoutine()
        gate = or_gate()
        register = routine.new_wires(gate.arity)
        routine.apply(init_state(alpha), register[0])
        routine.apply(init_state(beta), register[1])
        routine.apply(gate, register)
        self.job = routine.to_circ().to_job(
            nbshots=0,
            qubits=[gate.arity - 1]
        )
    def loss(self, guess, output):
        """
        Method for computing the loss of the OR gate
        Parameters
        ----------
        guess : list
            list with the inputs variables for the OR circuit.
        output : float
            Desired probability of |1> state for the OR gate.
        """
        alpha = guess[0]
        beta = guess[1]
        self.circuit(alpha, beta)
        pdf = solve_qjob(self.job, self.qpu)
        loss_ = (pdf[pdf["Int"] == 1]["Probability"] - output) ** 2
        return loss_
    def minimize(self):
        """
        Method for optimize the OR QLM quantum gate
        """
        bnds = ((0.0, 1.0) for i in self.guess_)
        res = minimize(
            self.loss,
            self.guess_,
            args=self.alpha_c,
            bounds=bnds
        )
        self.alpha_a = res.x[0]
        self.alpha_b = res.x[1]

class NOT(InferentialGate):
    """
    NOT gate class.
    Parameters
    ----------
    qpu : QLM qpu
        QLM qpu for solving the QLM quantum circuits
    alpha_c : float
        Desired probability of the |1> state for the NOT gate
    """

    def guess(self):
        """
        Method for initializing guess variables for minimization
        """
        self.alpha_a = 1.0 - self.alpha_c
        # Inital guess for optimization
        self.guess_ = np.array([self.alpha_a])


    def circuit(self, alpha, beta=None):
        """
        Method for building the corresponding quantum circuit for NOT gate
        Parameters
        ----------
        alpha : float
            Probability of |1> state for first input qubit
        """
        routine = qlm.QRoutine()
        gate = not_gate()
        register = routine.new_wires(gate.arity)
        routine.apply(init_state(alpha), register[0])
        routine.apply(gate, register)
        self.job = routine.to_circ().to_job(
            nbshots=0,
            qubits=[gate.arity - 1]
        )
    def loss(self, guess, output):
        """
        Method for computing the loss of the NOT gate
        Parameters
        ----------
        guess : list
            list with the inputs variables for the NOT circuit.
        output : float
            Desired probability of |1> state for the NOT gate.
        """
        alpha = guess[0]
        self.circuit(alpha)
        pdf = solve_qjob(self.job, self.qpu)
        loss_ = (pdf[pdf["Int"] == 1]["Probability"] - output) ** 2
        return loss_
    def minimize(self):
        """
        Method for optimize the NOT QLM quantum gate
        """
        bnds = ((0.0, 1.0) for i in self.guess_)
        res = minimize(
            self.loss,
            self.guess_,
            args=self.alpha_c,
            bounds=bnds
        )
        self.alpha_a = res.x[0]

class PRECISION(InferentialGate):
    """
    PRECISION gate class.
    Parameters
    ----------
    qpu : QLM qpu
        QLM qpu for solving the QLM quantum circuits
    alpha_c : float
        Desired probability of the |1> state for the PRECISION gate
    """
    def guess(self):
        """
        Method for initializing guess variables for minimization
        """
        self.guess_ = np.array([self.alpha_c])

    def circuit(self, alpha):
        """
        Method for building the corresponding quantum circuit for a fact
        with a fixed precision
        Parameters
        ----------
        alpha : float
            Precision of the fact
        """
        routine = qlm.QRoutine()
        gate = precision_gate(alpha)
        register = routine.new_wires(gate.arity)
        routine.apply(gate, register)
        self.job = routine.to_circ().to_job(
            nbshots=0,
            qubits=[gate.arity - 1]
        )
    def loss(self, guess, output):
        """
        Method for computing the loss of the PRECISION gate
        Parameters
        ----------
        guess : list
            list with the inputs variables for the OR circuit.
        output : float
            Desired probability of |1> state for the OR gate.
        """
        alpha = guess[0]
        self.circuit(alpha)
        pdf = solve_qjob(self.job, self.qpu)
        loss_ = (pdf[pdf["Int"] == 1]["Probability"] - output) ** 2
        return loss_

    def minimize(self):
        """
        Method for optimize the PRECISION QLM quantum gate.
        """
        bnds = ((0.0, 1.0) for i in self.guess_)
        res = minimize(
            self.loss,
            self.guess_,
            args=self.alpha_c,
            bounds=bnds
        )
        self.alpha_a = res.x[0]


class RULE(InferentialGate):
    """
    RULE gate class.
    Parameters
    ----------
    qpu : QLM qpu
        QLM qpu for solving the QLM quantum circuits
    alpha_c : float
        Desired probability of the |1> state for the RULE gate
    """
    def guess(self):
        """
        Method for initializing guess variables for minimization
        """
        self.alpha_b = random()
        self.alpha_a = self.alpha_c / self.alpha_b
        while self.alpha_a >= 1.0:
            self.alpha_b = random()
            self.alpha_a = self.alpha_c / self.alpha_b

        # Inital guess for optimization
        self.guess_ = np.array([self.alpha_a, self.alpha_b])

    def circuit(self, alpha, beta):
        """
        Method for building the corresponding quantum circuit for a rule
        with a fixed uncertainty
        Parameters
        ----------
        alpha : float
            Probability of |1> state for first input qubit
        beta : float
            Certainty of the rule
        """
        routine = qlm.QRoutine()
        gate = rule_gate(beta)
        register = routine.new_wires(gate.arity)
        routine.apply(init_state(alpha), register[0])
        routine.apply(gate, register)
        self.job = routine.to_circ().to_job(
            nbshots=0,
            qubits=[gate.arity - 1]
        )
    def loss(self, guess, output):
        """
        Method for computing the loss of the RULE gate
        Parameters
        ----------
        guess : list
            list with the inputs variables for the RULE circuit.
        output : float
            Desired probability of |1> state for the RULE gate.
        """
        alpha = guess[0]
        beta = guess[1]
        self.circuit(alpha, beta)
        pdf = solve_qjob(self.job, self.qpu)
        loss_ = (pdf[pdf["Int"] == 1]["Probability"] - output) ** 2
        return loss_

    def minimize(self):
        """
        Method for optimize the RULE QLM quantum gate.
        """
        bnds = ((0.0, 1.0) for i in self.guess_)
        res = minimize(
            self.loss,
            self.guess_,
            args=self.alpha_c,
            bounds=bnds
        )
        self.alpha_a = res.x[0]
        self.alpha_b = res.x[1]

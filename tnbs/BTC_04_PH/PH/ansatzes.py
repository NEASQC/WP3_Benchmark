"""
This module contains all functions needed for creating QLM
implementation for the ansatz of the Parent Hamiltonian paper:

    Fumiyoshi Kobayashi and Kosuke Mitarai and Keisuke Fujii
    Parent Hamiltonian as a benchmark problem for variational quantum
    eigensolvers
    Physical Review A, 105, 5, 2002
    https://doi.org/10.1103%2Fphysreva.105.052415

Authors: Gonzalo Ferro

"""

import pandas as pd
import numpy as  np
import qat.lang.AQASM as qlm
from qat.core import Result


def ansatz_qlm_01(nqubits=7, depth=3):
    """
    Implements QLM version of the Parent Hamiltonian Ansatz using
    parametric Circuit

    Parameters
    ----------

    nqubits: int
        number of qubits for the ansatz
    depth : int
        number of layers for the parametric circuit

    Returns
    _______

    qprog : QLM Program
        QLM program with the ansatz implementation in parametric format
    theta : list
        list with the name of the variables of the QLM Program
    """

    qprog = qlm.Program()
    qbits = qprog.qalloc(nqubits)

    #Parameters for the PQC
    theta = [
        qprog.new_var(float, "\\theta_{}".format(i)) for i in range(2*depth)
    ]

    for d_ in range(0, 2*depth, 2):
        for i in range(nqubits):
            qprog.apply(qlm.RX(theta[d_]), qbits[i])
        for i in range(nqubits-1):
            qprog.apply(qlm.Z.ctrl(), qbits[i], qbits[i+1])
        qprog.apply(qlm.Z.ctrl(), qbits[nqubits-1], qbits[0])
        for i in range(nqubits):
            qprog.apply(qlm.RZ(theta[d_+1]), qbits[i])
    theta = [th.name for th in theta]
    circuit = qprog.to_circ()
    return circuit

def ansatz_qlm_02(nqubits, depth=3):
    """
    Implements QLM version of the Parent Hamiltonian Ansatz using
    parametric Circuit

    Parameters
    ----------

    nqubits: int
        number of qubits for the ansatz
    depth : int
        number of layers for the parametric circuit

    Returns
    _______

    qprog : QLM Program
        QLM program with the ansatz implementation in parametric format
    theta : list
        list with the name of the variables of the QLM Program
    """

    qprog = qlm.Program()
    qbits = qprog.qalloc(nqubits)

    #Parameters for the PQC
    #theta = [
    #    qprog.new_var(float, "\\theta_{}".format(i)) for i in range(2*depth)
    #]

    theta = []
    indice = 0
    for d_ in range(0, 2*depth, 2):
        for i in range(nqubits):
            step = qprog.new_var(float, "\\theta_{}".format(indice))
            qprog.apply(qlm.RX(step), qbits[i])
            theta.append(step)
            indice = indice + 1
        for i in range(nqubits-1):
            qprog.apply(qlm.Z.ctrl(), qbits[i], qbits[i+1])
        qprog.apply(qlm.Z.ctrl(), qbits[nqubits-1], qbits[0])
        for i in range(nqubits):
            step = qprog.new_var(float, "\\theta_{}".format(indice))
            qprog.apply(qlm.RZ(step), qbits[i])
            indice = indice + 1
            theta.append(step)
    circuit = qprog.to_circ()
    return circuit

def proccess_qresults(result, qubits, complete=True):
    """
    Post Process a QLM results for creating a pandas DataFrame

    Parameters
    ----------

    result : QLM results from a QLM qpu.
        returned object from a qpu submit
    qubits : int
        number of qubits
    complete : bool
        for return the complete basis state.
    """

    # Process the results
    if complete:
        states = []
        list_int = []
        list_int_lsb = []
        for i in range(2**qubits):
            reversed_i = int("{:0{width}b}".format(i, width=qubits)[::-1], 2)
            list_int.append(reversed_i)
            list_int_lsb.append(i)
            states.append("|" + bin(i)[2:].zfill(qubits) + ">")
        probability = np.zeros(2**qubits)
        amplitude = np.zeros(2**qubits, dtype=np.complex_)
        for samples in result:
            probability[samples.state.lsb_int] = samples.probability
            amplitude[samples.state.lsb_int] = samples.amplitude

        pdf = pd.DataFrame(
            {
                "States": states,
                "Int_lsb": list_int_lsb,
                "Probability": probability,
                "Amplitude": amplitude,
                "Int": list_int,
            }
        )
    else:
        list_for_results = []
        for sample in result:
            list_for_results.append([
                sample.state, sample.state.lsb_int, sample.probability,
                sample.amplitude, sample.state.int,
            ])

        pdf = pd.DataFrame(
            list_for_results,
            columns=['States', "Int_lsb", "Probability", "Amplitude", "Int"]
        )
        pdf.sort_values(["Int_lsb"], inplace=True)
    return pdf


        


def solving_circuit(qlm_circuit, nqubit, qlm_qpu, reverse=True):
    """
    Solving a complete qlm circuit

    Parameters
    ----------

    qlm_circuit : QLM circuit
        qlm circuit to solve
    nqubit : int
        number of qubits of the input circuit
    qlm_qpu : QLM qpu
        QLM qpu for solving the circuit
    reverse : True
        This is for ordering the state from left to right
        If False the order will be form right to left

    Returns
    _______

    state : pandas DataFrame
        DataFrame with the complete simulation of the circuit
    """
    # Creating the qlm_job
    job = qlm_circuit.to_job()
    qlm_state = qlm_qpu.submit(job)
    if not isinstance(qlm_state, Result):
        qlm_state = qlm_state.join()
        # time_q_run = float(result.meta_data["simulation_time"])

    pdf_state = proccess_qresults(qlm_state, nqubit, True)
    # For keep the correct qubit order convention for following
    # computations
    if reverse:
        pdf_state.sort_values('Int', inplace=True)
    # A n-qubit-tensor is prefered for returning
    # state = np.array(pdf_state['Amplitude'])
    # mps_state = state.reshape(tuple(2 for i in range(nqubit)))
    return pdf_state

class SolveCircuit:

    def __init__(self, qlm_circuit, **kwargs):
        """

        Method for initializing the class

        """
        self.circuit = qlm_circuit
        self.nqubits = self.circuit.nbqbits
        self.parameter_names = self.circuit.get_variables()
        self.kwargs = kwargs
        self.par = kwargs.get("parameters", None)
        self._parameters = None
        self.parameters = self.par

        # Set the QPU to use
        self._qpu = None
        self.qpu = kwargs.get("qpu", None)
        # For Storing Results
        self.state = None

    @property
    def parameters(self):
        """
        creating parameters property
        """
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        """
        setter of the parameters property
        """

        if value is None:
            # Random Initialization will be used
            angles = list(2* np.pi * np.random.rand(len(self.parameter_names)))
            self._parameters = {v_ : angles[i_] for i_, v_ in enumerate(
                self.parameter_names)}
        else:
            if isinstance(value, (list, dict)) != True:
                text = "input must be a list, a dictionary or None"
                raise ValueError(text)

            if len(value) != len(self.parameter_names):
                text = "input must be a list, a dictionary or None"
                raise ValueError(text)


            if  isinstance(value, list):
                self._parameters = {v_ : value[i_] for i_, v_ in enumerate(
                    self.parameter_names)}

            if isinstance(value, dict):
                self._parameters = value

    @property
    def qpu(self):
        """
        creating qpu property
        """
        return self._qpu

    @qpu.setter
    def qpu(self, value):
        """
        setter of the qpu property
        """
        if value is None:
            error_text = "Please provide a QPU."
            raise ValueError(error_text)
        self._qpu = value

    def run(self):
        self.circuit = self.circuit(**self.parameters)
        self.state = solving_circuit(self.circuit, self.nqubits, self.qpu)


"""
Mandatory functions for performing classical QPE
Author: Gonzal Ferro
"""

import time
import numpy as np
import pandas as pd
import qat.lang.AQASM as qlm
from qat.core import Result
pd.options.display.float_format = "{:.6f}".format
np.set_printoptions(suppress=True)

def check_list_type(x_input, tipo):
    """Check if a list x_input is of type tipo
    Parameters
    ----------
    x_input : list
    tipo : data type
        it has to be understandable by numpy

    Returns
    ----------
    y_output : np.array
        numpy array of type tipo.
    """
    try:
        y_output = np.array(x_input).astype(tipo, casting="safe")
    except TypeError:
        exception = "Only a list/array of " + str(tipo) + " are aceptable types"
        raise Exception(exception) from TypeError
    return y_output

def get_results(
    quantum_object,
    linalg_qpu,
    shots: int = 0,
    qubits: list = None,
    complete: bool = False
):
    """
    Function for testing an input gate. This function creates the
    quantum program for an input gate, the correspondent circuit
    and job. Execute the job and gets the results

    Parameters
    ----------
    quantum_object : QLM Gate, Routine or Program
    linalg_qpu : QLM solver
    shots : int
        number of shots for the generated job.
        if 0 True probabilities will be computed
    qubits : list
        list with the qubits for doing the measurement when simulating
        if None measurement over all allocated qubits will be provided
    complete : bool
        for return the complete basis state. Useful when shots is not 0
        and all the posible basis states are necessary.

    Returns
    ----------
    pdf : pandas DataFrame
        DataFrame with the results of the simulation
    circuit : QLM circuit
    q_prog : QLM Program.
    job : QLM job
    """
    #print("BE AWARE!! linalg_qpu : {}".format(linalg_qpu))
    # if type(quantum_object) == qlm.Program:
    if isinstance(quantum_object, qlm.Program):
        q_prog = quantum_object
        arity = q_prog.qbit_count
    else:
        q_prog = create_qprogram(quantum_object)
        arity = quantum_object.arity

    if qubits is None:
        qubits = np.arange(arity, dtype=int)
    else:
        qubits = check_list_type(qubits, int)
    # circuit = q_prog.to_circ(submatrices_only=True)
    start = time.time()
    circuit = create_qcircuit(q_prog)
    end = time.time()
    time_q_circuit = end - start

    start = time.time()
    # job = circuit.to_job(nbshots=shots, qubits=qubits)
    job = create_qjob(circuit, shots=shots, qubits=qubits)
    end = time.time()
    time_q_job = end - start

    start = time.time()
    result = linalg_qpu.submit(job)
    if not isinstance(result, Result):
        result = result.join()
        # time_q_run = float(result.meta_data["simulation_time"])
        qpu_type = "QLM_QPU"
    else:
        qpu_type = "No QLM_QPU"
    end = time.time()
    time_q_run = end - start
    # Process the results
    start = time.time()
    pdf = proccess_qresults(result, qubits, complete=complete)
    end = time.time()
    time_post_proccess = end - start

    time_dict = {
        "time_q_circuit": time_q_circuit,
        "time_q_job": time_q_job,
        "time_q_run": time_q_run,
        "time_post_proccess": time_post_proccess,
    }
    pdf_time = pd.DataFrame([time_dict])
    pdf_time["time_total"] = pdf_time.sum(axis=1)
    pdf_time["qpu_type"] = qpu_type

    return pdf, circuit, q_prog, job


def create_qprogram(quantum_gate):
    """
    Creates a Quantum Program from an input qlm gate or routine

    Parameters
    ----------

    quantum_gate : QLM gate or QLM routine

    Returns
    ----------
    q_prog: QLM Program.
        Quantum Program from input QLM gate or routine
    """
    q_prog = qlm.Program()
    qbits = q_prog.qalloc(quantum_gate.arity)
    q_prog.apply(quantum_gate, qbits)
    return q_prog


def create_qcircuit(prog_q):
    """
    Given a QLM program creates a QLM circuit

    Parameters
    ----------

    prog_q : QLM QProgram

    Returns
    ----------

    circuit : QLM circuit
    """
    q_prog = prog_q
    circuit = q_prog.to_circ(submatrices_only=True)#, inline=True)
    return circuit


def create_qjob(circuit, shots=0, qubits=None):
    """
    Given a QLM circuit creates a QLM job

    Parameters
    ----------

    circuit : QLM circuit
    shots : int
        number of measurmentes
    qubits : list
        with the qubits to be measured

    Returns
    ----------

    job : QLM job
        job for submit to QLM QPU
    """
    dict_job = {"amp_threshold": 0.0}
    if qubits is None:
        job = circuit.to_job(nbshots=shots, **dict_job)
    else:
        if isinstance(qubits, (np.ndarray, list)):
            job = circuit.to_job(nbshots=shots, qubits=qubits, **dict_job)
        else:
            raise ValueError("qbits: sould be a list!!!")
    return job


def proccess_qresults(result, qubits, complete=False):
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
        for i in range(2**qubits.size):
            reversed_i = int("{:0{width}b}".format(i, width=qubits.size)[::-1], 2)
            list_int.append(reversed_i)
            list_int_lsb.append(i)
            states.append("|" + bin(i)[2:].zfill(qubits.size) + ">")

        probability = np.zeros(2**qubits.size)
        amplitude = np.zeros(2**qubits.size, dtype=np.complex_)
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


def load_qn_gate(qlm_gate, n_times):
    """
    Create an AbstractGate by applying an input gate n times

    Parameters
    ----------

    qlm_gate : QLM gate
        QLM gate that will be applied n times
    n_times : int
        number of times the qlm_gate will be applied

    """

    @qlm.build_gate(f"Q^{n_times}_{time.time_ns()}", [], arity=qlm_gate.arity)
    def q_n_gate():
        """
        Function generator for creating an AbstractGate for apply
        an input gate n times
        Returns
        ----------
        q_rout : quantum routine
            Routine for applying n times an input gate
        """
        q_rout = qlm.QRoutine()
        q_bits = q_rout.new_wires(qlm_gate.arity)
        for _ in range(n_times):
            q_rout.apply(qlm_gate, q_bits)
        return q_rout

    return q_n_gate()

class CQPE:
    """
    Class for using classical Quantum Phase Estimation, with inverse of
    Quantum Fourier Transformation.

    Parameters
    ----------

    kwars : dictionary
        dictionary that allows the configuration of the CQPE algorithm: \\
        Implemented keys:

        initial_state : QLM Program
            QLM Program with the initial Psi state over the
            Grover-like operator will be applied
            Only used if oracle is None
        unitary_operator : QLM gate or routine
            Grover-like operator which autovalues want to be calculated
            Only used if oracle is None
        cbits_number : int
            number of classical bits for phase estimation
        qpu : QLM solver
            solver for simulating the resulting circuits
        shots : int
            number of shots for quantum job. If 0 exact probabilities
            will be computed.
    """

    def __init__(self, **kwargs):
        """
        Method for initializing the class
        """

        # Setting attributes
        # In this case we load directly the initial state
        # and the grover operator
        self.initial_state = kwargs.get("initial_state", None)
        self.q_gate = kwargs.get("unitary_operator", None)
        if (self.initial_state is None) or (self.q_gate is None):
            text = "initial_state and grover keys should be provided"
            raise KeyError(text)

        # Number Of classical bits for estimating phase
        self.auxiliar_qbits_number = kwargs.get("auxiliar_qbits_number", 8)

        # Set the QPU to use
        self.linalg_qpu = kwargs.get("qpu", None)
        if self.linalg_qpu is None:
            print("Not QPU was provide. PyLinalg will be used")
            from qat.qpus import get_default_qpu
            self.linalg_qpu = get_default_qpu()

        self.shots = kwargs.get("shots", 10)
        self.complete = kwargs.get("complete", False)

        #Quantum Routine for QPE
        #Auxiliar qbits
        self.q_aux = None
        #Qubits rtegisters
        self.registers = None
        #List of ints with the position of the qbits for measuring
        self.meas_qbits = None
        #For storing results
        self.result = None
        #For storing qunatum times
        self.quantum_times = []
        #For storing the QPE routine
        self.circuit = None

    def run(self):
        """
        Creates the quantum phase estimation routine
        """
        qpe_routine = qlm.QRoutine()
        #Creates the qbits foe applying the operations
        self.registers = qpe_routine.new_wires(self.initial_state.arity)
        #Initializate the registers
        qpe_routine.apply(self.initial_state, self.registers)
        #Creates the auxiliary qbits for phase estimation
        self.q_aux = qpe_routine.new_wires(self.auxiliar_qbits_number)
        #Apply controlled Operator an increasing number of times
        for i, aux in enumerate(self.q_aux):
            #Apply Haddamard to all auxiliary qbits
            qpe_routine.apply(qlm.H, aux)
            #Power of the unitary operator depending of the position
            #of the auxiliary qbit.
            step_q_gate = load_qn_gate(self.q_gate, 2**i)
            #Controlled application of power of unitary operator
            qpe_routine.apply(step_q_gate.ctrl(), aux, self.registers)
        #Apply the QFT
        qpe_routine.apply(qlm.qftarith.QFT(len(self.q_aux)).dag(), self.q_aux)
        self.circuit = qpe_routine

        start = time.time()
        #Getting the result
        self.meas_qbits = [
            len(self.registers) + i for i, aux in enumerate(self.q_aux)]
        self.result, _, _, _ = get_results(
            self.circuit,
            linalg_qpu=self.linalg_qpu,
            shots=self.shots,
            qubits=self.meas_qbits,
            complete=self.complete
        )
        end = time.time()
        self.quantum_times.append(end-start)
        del self.result["Amplitude"]
        self.result["lambda"] = self.result["Int"] / (2**len(self.q_aux))


"""
Complete implementation of the Benchmark Test Case for Parent
Hamiltonian kernel.
Author: Gonzalo Ferro
"""
import logging
import time
import uuid
import sys
import numpy as np
import pandas as pd
from qat.core import Observable, Term

sys.path.append("../")
from PH.parent_hamiltonian import PH
from PH.ansatzes import SolveCircuit, ansatz_selector

logging.basicConfig(
    format='%(asctime)s-%(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO
    #level=logging.DEBUG
)
logger = logging.getLogger('__name__')


def get_qpu(qpu=None):
    """
    Function for selecting solver.

    Parameters
    ----------

    qpu : str
        * qlmass: for trying to use QLM as a Service connection to CESGA QLM
        * python: for using PyLinalg simulator.
        * c: for using CLinalg simulator
        * mps: for using mps

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
    elif qpu == "mps":
        try:
            from qlmaas.qpus import MPS
            linalg_qpu = MPS(lnnize=True)
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


def ph_btc(**kwargs):
    ansatz = kwargs.get("ansatz", None)
    nqubits = kwargs.get("nqubits", None)
    depth = kwargs.get("depth", None)
    qpu_ansatz = kwargs.get("qpu_ansatz", None)
    parameters = kwargs.get("parameters", None)
    qpu_ph = kwargs.get("qpu_ph", None)
    nb_shots = kwargs.get("nb_shots", 0)
    save = kwargs.get("save", False)
    folder = kwargs.get("folder", './')
    ansatz_conf = {
        "nqubits": nqubits, "depth": depth
    }
    ansatz_circuit = ansatz_selector(ansatz, **ansatz_conf)
    solve_conf = {
        "qpu" : qpu_ansatz, "parameters": parameters

    }
    sol_ansatz = SolveCircuit(ansatz_circuit, **solve_conf)
    logger.info('Solving ansatz')
    tick = time.time()
    sol_ansatz.run()
    # Now the circuit need to have the parameters
    ansatz_circuit = sol_ansatz.circuit
    amplitudes = list(sol_ansatz.state['Amplitude'])
    ph_ansatz = PH(amplitudes)
    logger.info('Computing Local Parent Hamiltonian')
    ph_ansatz.local_ph()
    pauli_coefs = ph_ansatz.pauli_coeficients
    pauli_strings = ph_ansatz.pauli_matrices
    affected_qubits = ph_ansatz.qubits_list
    pauli_df = pd.DataFrame(
        [pauli_coefs, pauli_strings, affected_qubits],
        index=['PauliCoefficients', 'PauliStrings', 'Qbits']).T
    angles = [k for k, v in sol_ansatz.parameters.items()]
    values = [v for k, v in sol_ansatz.parameters.items()]
    parameter_ansatz = pd.DataFrame(
        [angles, values],
        index=['key', 'value']).T
    filename_base = str(uuid.uuid1())
    if save:
        pauli_df.to_csv(folder + filename_base+'_pauli.csv', sep=';')
        parameter_ansatz.to_csv(
            folder + filename_base+'_parameters.csv', sep=';')
    # Creating Pauli Terms
    terms = [Term(coef, ps, qb) for coef, ps, qb in \
        zip(pauli_coefs, pauli_strings, affected_qubits)]
    # Creates the atos myqlm observable
    observable = Observable(nqbits=nqubits, pauli_terms=terms)
    # Creates the job
    job_ansatz = ansatz_circuit.to_job(
        'OBS', observable=observable, nbshots=nb_shots)
    logger.info('Execution Local Parent Hamiltonian')
    tick_q = time.time()
    # Quantum Routine
    gse = qpu_ph.submit(job_ansatz)
    gse = gse.value
    tock = time.time()
    quantum_time = tock - tick_q
    elapsed_time = tock - tick
    text = ['gse', 'elapsed_time', 'quantum_time']
    res = pd.DataFrame(
        [gse, elapsed_time, quantum_time],
        index=text
    ).T
    if save:
        res.to_csv(folder + filename_base+'_result.csv', sep=';')
    return res

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-nqubits",
        dest="nqubits",
        type=int,
        help="Number of qbits for the ansatz.",
        default=None,
    )
    parser.add_argument(
        "-depth",
        dest="depth",
        type=int,
        help="Depth for ansatz.",
        default=None,
    )
    parser.add_argument(
        "-ansatz",
        dest="ansatz",
        type=str,
        help="Ansatz type: simple01, simple02, lda or hwe.",
        default=None,
    )
    #QPU argument
    parser.add_argument(
        "-qpu_ansatz",
        dest="qpu_ansatz",
        type=str,
        default=None,
        help="QPU for ansatz simulation: [qlmass, python, c, mps]",
    )
    parser.add_argument(
        "-qpu_ph",
        dest="qpu_ph",
        type=str,
        default=None,
        help="QPU for parent hamiltonian simulation: [qlmass, python, c]",
    )
    parser.add_argument(
        "--save",
        dest="save",
        default=False,
        action="store_true",
        help="For storing results",
    )
    parser.add_argument(
        "-folder",
        dest="folder",
        type=str,
        default="./",
        help="Path for stroing results",
    )
    args = parser.parse_args()
    print(args)
    dict_ph = vars(args)
    dict_ph.update({"qpu_ansatz": get_qpu(dict_ph['qpu_ansatz'])})
    dict_ph.update({"qpu_ph":  get_qpu(dict_ph['qpu_ph'])})
    print(dict_ph)
    ph_btc(**dict_ph)

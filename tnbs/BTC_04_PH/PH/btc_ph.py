"""
Complete implementation of the Benchmark Test Case for Parent
Hamiltonian kernel.
Author: Gonzalo Ferro
"""
import logging
import time
import numpy as np
import pandas as pd

from parent_hamiltonian import PH
from ansatzes import SolveCircuit, ansatz_selector, angles_ansatz01
from execution_ph import PH_EXE
from utils import create_folder, get_qpu

logger = logging.getLogger('__name__')

def complete_ph(**kwargs):
    """
    Execute a complete Bechmark Test Case for PH Kernel
    """

    # Getting Configuration
    ansatz = kwargs.get("ansatz", None)
    nqubits = kwargs.get("nqubits", None)
    depth = kwargs.get("depth", None)
    qpu_ansatz = kwargs.get("qpu_ansatz", None)
    t_invariant = kwargs.get("t_inv", None)
    qpu_ph = kwargs.get("qpu_ph", None)
    truncation = kwargs.get("truncation", None)
    nb_shots = kwargs.get("nb_shots", 0)
    # parameters = kwargs.get("parameters", None)
    save = kwargs.get("save", False)
    folder = kwargs.get("folder", "./")
    filename = kwargs.get("filename", None)
    pdf_info = pd.DataFrame(kwargs, index=[0])

    # Ansatz Configuration
    ansatz_conf = {
        'nqubits' : nqubits,
        'depth' : depth
    }

    # Create Ansatz Circuit
    tick = time.time()
    logger.info("Creating ansatz circuit")
    circuit = ansatz_selector(ansatz, **ansatz_conf)
    tack = time.time()
    create_ansatz_time = tack - tick
    logger.info("Created ansatz circuit in: %s", create_ansatz_time)

    if ansatz == "simple01":
        #If ansatz is simple we use fixed angles
        circuit, pdf_parameters = angles_ansatz01(circuit)
    else:
        # For other ansatzes we use random parameters
        parameters = {v_ : 2 * np.pi * np.random.rand() for i_, v_ in enumerate(
            circuit.get_variables())}
        # Create the DataFrame with the info
        angles = [k for k, v in parameters.items()]
        values = [v for k, v in parameters.items()]
        # create pdf
        pdf_parameters = pd.DataFrame(
            [angles, values],
            index=['key', 'value']).T
        circuit, _ = angles_ansatz01(circuit, pdf_parameters)

    # Solving Ansatz
    logger.info("Solving ansatz circuit")
    solve_conf = {
        "qpu" : qpu_ansatz,
        "parameters" : pdf_parameters,
        "filename": folder + filename,
        "save": save
    }
    solv_ansatz = SolveCircuit(circuit, **solve_conf)
    solv_ansatz.run()
    solve_ansatz_time = solv_ansatz.solve_ansatz_time
    logger.info("Solved ansatz circuit in: %s", solve_ansatz_time)

    # Create PH
    logger.info("Computing Local Parent Hamiltonian")
    amplitudes = list(solv_ansatz.state["Amplitude"])
    ph_conf = {
        "filename": folder + filename,
        "save": save
    }
    ph_object = PH(amplitudes, t_invariant, **ph_conf)
    ph_object.local_ph()
    ph_time = ph_object.ph_time
    logger.info("Computed Local Parent Hamiltonian in: %s", ph_time)

    # Executing VQE step
    logger.info("Executing VQE step")
    vqe_conf = {
        "qpu" : qpu_ph,
        "nb_shots": nb_shots,
        "truncation": truncation,
        "filename": folder + filename,
        "save": save
    }
    ansatz_circuit = solv_ansatz.circuit
    pauli_ph = ph_object.pauli_pdf
    nqubits = ansatz_conf["nqubits"]
    exe_ph = PH_EXE(ansatz_circuit, pauli_ph, nqubits, **vqe_conf)
    exe_ph.run()
    pdf_info["create_ansatz_time"] = create_ansatz_time
    pdf_info["solve_ansatz_time"] = create_ansatz_time
    pdf_info["ph_time"] = ph_time

    pdf = pd.concat([pdf_info, exe_ph.pdf_result], axis=1)
    times = [
        "create_ansatz_time", "solve_ansatz_time", "ph_time",
        "observable_time", "quantum_time"
    ]

    elapsed_time = pdf[times].sum(axis=1)
    pdf["elapsed_time"] = elapsed_time
    logger.info("Benchmark Test Case computed in %f", elapsed_time)
    return pdf

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
        "-nb_shots",
        dest="nb_shots",
        type=int,
        help="Number of shots",
        default=0,
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
    parser.add_argument(
        "-truncation",
        dest="truncation",
        type=int,
        help="Truncation for Pauli coeficients.",
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
        "-folder",
        dest="folder",
        type=str,
        default="./",
        help="Path for stroing results",
    )
    parser.add_argument(
        "--save",
        dest="save",
        default=False,
        action="store_true",
        help="For storing results",
    )
    parser.add_argument(
        "--t_inv",
        dest="t_inv",
        default=False,
        action="store_true",
        help="Setting translational invariant of the ansatz",
    )
    #Execution argument
    parser.add_argument(
        "--exe",
        dest="execution",
        default=False,
        action="store_true",
        help="For executing program",
    )
    args = parser.parse_args()
    dict_ph = vars(args)
    dict_ph.update({"qpu_ansatz": get_qpu(dict_ph["qpu_ansatz"])})
    dict_ph.update({"qpu_ph":  get_qpu(dict_ph["qpu_ph"])})
    if dict_ph["save"]:
        dict_ph.update({"folder": create_folder(dict_ph["folder"])})
    filename = "ansatz_{}_depth_{}_nqubits_{}".format(
        args.ansatz, args.depth, args.nqubits)
    dict_ph.update({"filename":filename})
    print(dict_ph)
    if args.execution:
        result = complete_ph(**dict_ph)
        print(result)

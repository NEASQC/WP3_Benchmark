"""
For executing a VQE quantum step of a ansatz and a given
Parent Hamiltonian.
In this case the input are a folder with the following pattern name:
    ansatz_{}_nqubits_{}_depth_{}_qpu_ansatz_{}
Inside the folder it is expected that following pattern files can be
found:
    {}_parameters.csv
    {}_pauli.csv

Author: Gonzalo Ferro
"""

import logging
import ast
import pandas as pd
from utils_ph import get_info_basefn, get_qpu
from ansatzes import ansatz_selector, angles_ansatz01
from vqe_step import PH_EXE

logger = logging.getLogger("__name__")

def run_ph_execution(**configuration):
    """
    Given an ansatz circuit, the parameters and the Pauli decomposition
    of the corresponding local PH executes a VQE step for computing
    the energy of the ansatz under the Hamiltonian that MUST BE near 0
    Given a Folder with following pattern:
        * ansatz_{}_nqubits_{}_depth_{}_qpu_ansatz_{}
    That contains files with following patter:
        * {}_parameters.csv
        * {}_pauli.csv
    Workflow:
        * Create QLM circuit using the ansatz type readed from the folder
        * Loading parameters for the circuit from: {}_parameters.csv
        * Loading Pauli Decomposition from: {}_pauli.csv
        * Executes VQE step.
        * Stores the result of the execution: {}_phexe.csv
    """

    logger.info("Creating ansatz circuit")
    base_fn = configuration["base_fn"]
    print(base_fn)
    depth, nqubits, ansatz = get_info_basefn(base_fn)
    ansatz_conf = {
        "nqubits" :nqubits,
        "depth" : depth,
    }
    circuit = ansatz_selector(ansatz, **ansatz_conf)

    logger.info("Loading Parameters")
    parameters_pdf = pd.read_csv(
        base_fn + "_parameters.csv", sep=";", index_col=0)
    # Formating Parameters
    circuit, _ = angles_ansatz01(circuit, parameters_pdf)
    #from qat.core.console import display
    #display(circuit)

    # Loading PH Pauli decomposition
    logger.info("Loading PH Pauli decomposition")
    # Loading Pauli
    pauli_pdf = pd.read_csv(
        base_fn + "_pauli.csv", sep=";", index_col=0)
    affected_qubits = [ast.literal_eval(i_) for i_ in list(pauli_pdf["Qbits"])]
    pauli_pdf["Qbits"] = affected_qubits

    # Executing VQE step
    logger.info("Executing VQE step")
    vqe_conf = {
        "qpu" : get_qpu(configuration["qpu_ph"]),
        "nb_shots": configuration["nb_shots"],
        "truncation": configuration["truncation"],
        "t_inv": configuration["t_inv"],
        "filename": base_fn,
        "save": configuration["save"],
    }
    exe_ph = PH_EXE(circuit, pauli_pdf, nqubits, **vqe_conf)
    exe_ph.run()
    return exe_ph.pdf

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        format='%(asctime)s-%(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO
        #level=logging.DEBUG
    )
    logger = logging.getLogger('__name__')
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-basefn",
        dest="base_fn",
        type=str,
        default="",
        help="Base Filename for Loading Files",
    )
    parser.add_argument(
        "-nb_shots",
        dest="nb_shots",
        type=int,
        help="Number of shots",
        default=0,
    )
    parser.add_argument(
        "-truncation",
        dest="truncation",
        type=int,
        help="Truncation for Pauli coeficients.",
        default=None,
    )
    parser.add_argument(
        "-qpu_ph",
        dest="qpu_ph",
        type=str,
        default=None,
        help="QPU for parent hamiltonian simulation: [qlmass, python, c]",
    )
    parser.add_argument(
        "--t_inv",
        dest="t_inv",
        default=False,
        action="store_true",
        help="Setting translational invariant of the ansatz",
    )
    parser.add_argument(
        "--save",
        dest="save",
        default=False,
        action="store_true",
        help="For storing results",
    )
    args = parser.parse_args()

    print(run_ph_execution(**vars(args)))

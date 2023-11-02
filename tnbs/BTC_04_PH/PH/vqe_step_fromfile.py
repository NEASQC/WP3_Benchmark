"""
For executing a VQE quantum step of a ansatz and a given
Parent Hamiltonian.
Here the input is a base file name that should have following pattern:
    ansatz_{}_nqubits_{}_depth_{}_qpu_ansatz_{}
From this name following information should be extracted:
* ansatz_{}: the name of the ansatz
* nqubits_{}: the number of qubits for the ansatz
* depth_{}: the depth of the ansatz
* qpu_ansatz_{}: the qpu used for solving the ansatz

For the input base file name following files has to exist:
* {}_parameters.csv
* {}_pauli.csv

Here alwways talk about complete file names. The pattern can be found
in the bare name of the file, or in the folder that contain the file.
Example of valid names:
* ansatz_simple01_nqubits_27_depth_4_qpu_ansatz_qat_qlmass/8b961e30-5fc2-11ee-b12a-080038bfd786
* ansatz_simple01_nqubits_16_depth_4_qpu_c.csv

Author: Gonzalo Ferro
"""

import logging
import ast
import re
import pandas as pd
from utils_ph import get_qpu
from ansatzes import ansatz_selector, angles_ansatz01
from vqe_step import PH_EXE

logger = logging.getLogger("__name__")

def get_info_basefn(base_fn):
    depth = int(re.findall(r"depth_(.*)_qpu", base_fn)[0])
    nqubits = int(re.findall(r"nqubits_(.*)_depth_", base_fn)[0])
    ansatz = re.findall(r"ansatz_(.*)_nqubits", base_fn)[0]
    return depth, nqubits, ansatz

def run_ph_execution(**configuration):
    """
    Given an ansatz circuit, the parameters and the Pauli decomposition
    of the corresponding local PH executes a VQE step for computing
    the energy of the ansatz under the Hamiltonian that MUST BE near 0
    Given an input base_fn that MUST have following pattern:
        * base_fn = ansatz_{}_nqubits_{}_depth_{}_qpu_ansatz_{}
    Additionally folowing files MUST exist:
        * {base_fn}_parameters.csv
        * {base_fn}_pauli.csv
    The functions gets the information about: ansatz, nqubits and depth
    and executes the following Workflow:
        1. Create QLM circuit using the ansatz type readed from the folder
        2 Loading parameters for the circuit from: {}_parameters.csv
        3. Loading Pauli Decomposition from: {}_pauli.csv
        4. Executes VQE step.
    If save is True the result of the execution is stored as:
        * {base_fn}_phexe.csv
    """

    # 1. Create QLM circuit using the ansatz type readed from the folder
    logger.info("Creating ansatz circuit")
    base_fn = configuration["base_fn"]
    depth, nqubits, ansatz = get_info_basefn(base_fn)
    text = "ansatz: {0}, nqubits: {1} depth: {2}".format(ansatz, nqubits, depth)
    logger.debug(text)
    ansatz_conf = {
        "nqubits" :nqubits,
        "depth" : depth,
    }
    circuit = ansatz_selector(ansatz, **ansatz_conf)

    # 2 Loading parameters for the circuit from: {}_parameters.csv
    text = "Loading Parameters from: {}".format(base_fn + "_parameters.csv")
    logger.info(text)
    parameters_pdf = pd.read_csv(
        base_fn + "_parameters.csv", sep=";", index_col=0)
    # Formating Parameters
    circuit, _ = angles_ansatz01(circuit, parameters_pdf)
    # from qat.core.console import display
    # display(circuit)

    # 3. Loading Pauli Decomposition from: {}_pauli.csv
    text = "Loading PH Pauli decomposition from: {}".format(
        base_fn + "_parameters.csv")
    logger.info(text)
    # Loading Pauli
    pauli_pdf = pd.read_csv(
        base_fn + "_pauli.csv", sep=";", index_col=0)
    affected_qubits = [ast.literal_eval(i_) for i_ in list(pauli_pdf["Qbits"])]
    pauli_pdf["Qbits"] = affected_qubits

    # 4. Executes VQE step.
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
        #level=logging.INFO
        level=logging.DEBUG
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

"""
For executing a VQE quantum step of a ansatz and a given
Parent Hamiltonian. Allows execution of the configuration
files for the BTC (BTC_04_PH/configuration_files/ folder)
In this case the inputs are:
1. Number of the qubits of the ansatz (nqubits, keyword).
2. Depth of the ansatz (depth, keyword).
3. base_fn: folder where the csv with parameters and pauli decomposition
    is stored. Patterns for the name of these files:
    * nqubits_{}_depth_{}_parameters.csv
    * nqubits_{}_depth_{}_pauli.csv
The keytowrd inputs (nqubits, depth) are used for building the number of
the parameters and pauli decomposition files.

Author: Gonzalo Ferro
"""

import sys
import logging
import ast
import pandas as pd
sys.path.append("../../../")
from qpu.select_qpu import select_qpu
from PH.ansatzes.ansatzes import ansatz_selector, angles_ansatz01
from PH.ph_step_exe.vqe_step import PH_EXE

logger = logging.getLogger("__name__")


def run_ph_execution(**configuration):
    """
    Given an ansatz circuit, the parameters and the Pauli decomposition
    of the corresponding local PH executes a VQE step for computing
    the energy of the ansatz under the Hamiltonian that MUST BE near 0
    Workflow:
        * Getting the nqubits and depth from keyword arguments.
        * Using them and the base_fn following file names are created:
        base_fn + nqubits_{}_depth_{}_parameters.csv
        base_fn + nqubits_{}_depth_{}_pauli.csv
        * The Parameter are loading from parameters file.
        * The QLM circuit is created (simplest ansatz) using
        nqubits and depth.
        * Parameters are loading in QLM circuit.
        * The Pauli decomposition is loaded from pauli files.
        * Executes VQE step.
    """

    logger.info("Creating ansatz circuit")
    nqubits = configuration["nqubits"]
    depth = configuration["depth"]
    ansatz_conf = {
        "nqubits" :nqubits,
        "depth" : depth,
    }
    circuit = ansatz_selector("simple01", **ansatz_conf)

    logger.info("Loading Parameters")
    base_fn = configuration["base_fn"] + "/nqubits_{}_depth_{}".format(
        str(nqubits).zfill(2), depth)
    print(base_fn)
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
        "qpu" : configuration["qpu"],
        "nb_shots": configuration["nb_shots"],
        "truncation": configuration["truncation"],
        "t_inv": configuration["t_inv"],
        "filename": base_fn,
        "save": configuration["save"],
    }
    exe_ph = PH_EXE(circuit, pauli_pdf, nqubits, **vqe_conf)
    exe_ph.run()
    return exe_ph.pdf, circuit


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
    config = vars(args)
    qpu_config = {"qpu_type": args.qpu_ph}
    config.update({"qpu": select_qpu(qpu_config)})

    print(run_ph_execution(**vars(args)))

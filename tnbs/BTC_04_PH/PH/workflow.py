"""
Complete WorkFlow of a PH VQE
Author: Gonzalo Ferro
"""

import logging
import pandas as pd
from ansatzes import run_ansatz
from parent_hamiltonian import PH
from vqe_step import PH_EXE
import sys
sys.path.append("../")
from get_qpu import get_qpu

logger = logging.getLogger('__name__')

def workflow(**configuration):

    logger.info("Solving ansatz circuit")
    ansatz_dict = run_ansatz(**configuration)
    state = ansatz_dict["state"]
    circuit = ansatz_dict["circuit"]
    solve_ansatz_time = ansatz_dict["solve_ansatz_time"]
    filename = ansatz_dict["filename"]

    logger.info("Computing Local Parent Hamiltonian")
    ph_conf = {
        "filename": filename,
        "save": configuration["save"]
    }
    t_inv = configuration["t_inv"]
    ph_object = PH(list(state["Amplitude"]), t_inv, **ph_conf)
    ph_object.local_ph()
    ph_time = ph_object.ph_time

    logger.info("Executing VQE step")
    vqe_conf = {
        "t_inv" : t_inv,
        "qpu" : get_qpu(configuration["qpu_ph"]),
        "nb_shots": configuration["nb_shots"],
        "truncation": configuration["truncation"],
        "filename": filename,
        "save": configuration["save"]
    }
    pauli_ph = ph_object.pauli_pdf
    nqubits = configuration["nqubits"]
    exe_ph = PH_EXE(circuit, pauli_ph, nqubits, **vqe_conf)
    exe_ph.run()
    pdf_info = pd.DataFrame(configuration, index=[0])
    pdf_info["solve_ansatz_time"] = solve_ansatz_time
    pdf_info["ph_time"] = ph_time
    pdf_info = pd.concat([pdf_info, exe_ph.pdf_result], axis=1)
    if configuration["save"]:
        pdf_info.to_csv(filename + "_workflow.csv", sep=";")
    return pdf_info

if __name__ == "__main__":
    # For sending ansatzes to QLM
    import argparse
    logging.basicConfig(
        format='%(asctime)s-%(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO
        #level=logging.DEBUG
    )
    logger = logging.getLogger('__name__')

    parser = argparse.ArgumentParser()

    # Ansatz Configuration
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
    parser.add_argument(
        "-qpu_ansatz",
        dest="qpu_ansatz",
        type=str,
        default=None,
        help="QPU for ansatz simulation: [qlmass, python, c, mps]",
    )

    # Parent Hamiltonian
    parser.add_argument(
        "--t_inv",
        dest="t_inv",
        default=False,
        action="store_true",
        help="Setting translational invariant of the ansatz",
    )

    # VQE step execution
    parser.add_argument(
        "-truncation",
        dest="truncation",
        type=int,
        help="Truncation for Pauli coeficients.",
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
        help="Path for storing results",
    )
    parser.add_argument(
        "--save",
        dest="save",
        default=False,
        action="store_true",
        help="For storing results",
    )
    args = parser.parse_args()
    workflow(**vars(args))

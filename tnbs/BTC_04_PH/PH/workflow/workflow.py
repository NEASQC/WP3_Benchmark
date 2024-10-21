"""
Complete WorkFlow of a PH VQE
Author: Gonzalo Ferro
"""

import sys
import logging
import pandas as pd
sys.path.append("../../")
from PH.ansatzes.ansatzes import run_ansatz
from PH.parent_hamiltonian.parent_hamiltonian import PH
from PH.ph_step_exe.vqe_step import PH_EXE

logger = logging.getLogger('__name__')

def workflow(**configuration):
    """
    Executes complete Workflow:
    1. Create ansatz circuit and solve it.
    2. Computes Parent Hamiltonian
    3. Solve the ansatz under the Parent Hamiltonian
    """

    logger.info("Solving ansatz circuit")
    ansatz_conf = {
        "nqubits" : configuration["nqubits"],
        "depth": configuration["depth"],
        "ansatz" : configuration["ansatz"],
        "qpu" : configuration["ansatz_qpu"],
        "qpu_ansatz" : configuration["qpu_ansatz"],
        "save": configuration["save"],
        "folder": configuration["folder"],
    }
    print(ansatz_conf)
    ansatz_dict = run_ansatz(**ansatz_conf)
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
        "qpu" : configuration["ph_qpu"],
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
    import json
    sys.path.append("../../../")
    from qpu.select_qpu import select_qpu
    from qpu.benchmark_utils import combination_for_list
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
        help="QPU for parent hamiltonian simulation: " +
            "c, python, linalg, mps, qlmass_linalg, qlmass_mps",
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
        default="../qpu/qpu_ideal.json",
        help="JSON with the qpu configuration for ground state computation",
    )
    parser.add_argument(
        "-qpu_id",
        dest="qpu_id",
        type=int,
        default=None,
        help="Select a QPU for ground state computation from a JSON file",
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
    parser.add_argument(
        "--print",
        dest="print",
        default=False,
        action="store_true",
        help="For printing the selected QPU configuration."
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
    config = vars(args)
    # First set the qpu for the ansatz. No noisy simulation allowed
    qpu_config = {"qpu_type": args.qpu_ansatz}
    config.update({"ansatz_qpu": select_qpu(qpu_config)})
    config.update({"qpu_ansatz": args.qpu_ansatz})
    # Second set the qpu for solving the ground state energy
    with open(args.qpu_ph) as json_file:
        qpu_cfg = json.load(json_file)
    qpu_list = combination_for_list(qpu_cfg)
    if args.print:
        if args.qpu_id is not None:
            print(qpu_list[args.qpu_id])
        else:
            print("All possible QPUS will be printed: ")
            print(qpu_list)
    if args.execution:
        if args.qpu_id is not None:
            config.update({"ph_qpu": select_qpu(qpu_list[args.qpu_id])})
            print(workflow(**config))
        else:
            raise ValueError(
                "BE AWARE. For execution the -qpu_id is Mandatory"
            )

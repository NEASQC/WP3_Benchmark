"""
Parent Hamiltonian ansatz using MPS
"""

import sys
import numpy as np
import pandas as pd
import logging
sys.path.append("../../")
from PH.utils.utils_ph import create_folder
from PH.mps.mps import apply_local_gate, apply_2qubit_gates, compose_mps, \
    contract_indices_one_tensor
import PH.mps.gates_mps as gt
logger = logging.getLogger('__name__')

def get_angles(depth):
    """
    Setting the angles of the ansatz following TNBS procedure

    Parameters
    ----------

    depth : int
        Integer with the depth of the ansatz

    Returns
    _______

    mps_ : list
        list with the angles for the circuit
    """

    theta = np.pi/4.0
    delta_theta = theta / (depth + 1)
    angles = []
    for i in range(depth):
        angles.append([(2 * i + 1) * delta_theta, (2 * i + 2) * delta_theta])
    return angles

def ansatz_mps(nqubits, depth, angles, truncate=True, t_v=None):
    """
    Creates the MPS solution of the ansatz for TNBS BTC_04_PH benchmark

    Parameters
    ----------

    nqubits : int
        Number of qubits for the ansatz
    depth : int
        Depth for the ansatz
    truncate : bool
        Boolean variable for truncate SVD of the 2 qubits ansatz. If
        truncate is True and t_v is None precision of the float will be
        used as truncation value
    t_v: float
        Float for tuncating the SVD of the 2 qubits gates. Only valid if
        truncate is True,

    Returns
    _______

    mps_ : list
        list with the angles for the circuit
    """
    # Intitial State
    zeroket = np.zeros((1, 2, 1))
    zeroket[0][0][0] = 1
    zeroket = zeroket.astype(complex)
    #Initial State
    mps_ = [zeroket] * nqubits
    for depth_ in range(depth):
        # First Layer
        gates = [gt.x_rotation(angles[depth_][0]) for i in mps_]
        mps_ = apply_local_gate(mps_, gates)
        ent_gates = [gt.controlz() for i in mps_]
        mps_ = apply_2qubit_gates(mps_, ent_gates, truncate=truncate, t_v=t_v)
        gates = [gt.z_rotation(angles[depth_][1]) for i in mps_]
        mps_ = apply_local_gate(mps_, gates)
    return mps_

def state_computation(**configuration):
    """
    Computes State of the TNBS BTC_04_PH ansatz using MPS.

    Parameters
    ----------

    configuration MUST have following keyword arguments:

    nqubits : int
        Number of qubits for the ansatz
    depth : int
        Depth for the ansatz
    truncate : bool
        Boolean variable for truncate SVD of the 2 qubits ansatz. If
        truncate is True and t_v is None precision of the float will be
        used as truncation value
    t_v: float
        Float for tuncating the SVD of the 2 qubits gates. Only valid if
        truncate is True,
    save : bool
        For saving the resulting state
    folder : string
        Folder for saving the results

    Returns
    _______

    pdf : pandas DataFrame
        DataFrame with the result of the state of the circuit


    """
    # Ansatz configuration
    nqubits = configuration["nqubits"]
    depth = configuration["depth"]
    truncate = configuration["truncate"]
    t_v = configuration["t_v"]
    # PH configuration
    save = configuration["save"]
    folder_name = configuration["folder"]

    if save:
        folder_name = create_folder(folder_name)
        base_fn = folder_name + "ansatz_simple01_nqubits_{}_depth_{}".format(
            str(nqubits).zfill(2), depth)
    # Build Angles
    angles = get_angles(depth)
    list_angles = []
    for angle in angles:
        list_angles = list_angles + angle
    param_names = ["\\theta_{}".format(i) for i, _ in enumerate(list_angles)]
    pdf_angles = pd.DataFrame(
        [param_names, list_angles], index=["key", "value"]).T
    if save:
        pdf_angles.to_csv(base_fn + "_parameters.csv", sep=";")
    # Build MPS of the ansatz
    mps = ansatz_mps(nqubits, depth, angles, truncate, t_v)
    tensor = compose_mps(mps)
    state = contract_indices_one_tensor(tensor, [(0, tensor.ndim-1)])
    state = state.reshape(np.prod(state.shape))
    pdf = pd.DataFrame(state, columns=["Amplitude"])
    state_name = ["|" + bin(i)[2:].zfill(nqubits) + ">" \
        for i in range(2**nqubits)]
    pdf["state"] = state_name
    if save:
        pdf.to_csv(base_fn + "_state.csv", sep=";")
    return pdf

if __name__ == "__main__":
    import logging
    #logging.basicConfig(
    #    format='%(asctime)s-%(levelname)s: %(message)s',
    #    datefmt='%m/%d/%Y %I:%M:%S %p',
    #    level=logging.INFO
    #    #level=logging.DEBUG
    #)
    logger = logging.getLogger('__name__')
    # Given a state Compute its Parent Hamiltonian
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
        "--truncate",
        dest="truncate",
        default=False,
        action="store_true",
        help="Truncating the SVD. Float resolution will be used",
    )
    parser.add_argument(
        "-tv",
        dest="t_v",
        type=float,
        help="Truncation Value for the SVD. Only valid for  --truncate",
        default=None,
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
        default="",
        help="Folder for Storing Results",
    )
    parser.add_argument(
        "--print",
        dest="print",
        default=False,
        action="store_true",
        help="For printing the AE algorihtm configuration."
    )
    parser.add_argument(
        "--exe",
        dest="execution",
        default=False,
        action="store_true",
        help="For executing program",
    )
    args = parser.parse_args()
    if args.print:
        print(args)
    if args.execution:
        pdf = state_computation(**vars(args))
        print(pdf)

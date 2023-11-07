"""
Complete implementation of the Parent Hamiltonian following:
    Fumiyoshi Kobayashi and Kosuke Mitarai and Keisuke Fujii
    Parent Hamiltonian as a benchmark problem for variational quantum
    eigensolvers
    Physical Review A, 105, 5, 2002
    https://doi.org/10.1103%2Fphysreva.105.052415

Author: Gonzalo Ferro
"""
import logging
import time
import numpy as np
import pandas as pd
from scipy import linalg

from contractions import contract_indices
from pauli import pauli_decomposition
from ansatz_mps import ansatz_mps, get_angles
from utils_ph import create_folder
from mps import contraction_physical_legs, mpo_contraction
logger = logging.getLogger('__name__')

def reduced_rho_mps(mps, free_indices, contraction_indices):
    """
    Computes reduced density matrix by contracting first the MPS
    with NON contracted legs. Then compute the contraction of the
    MPS with contracted legs. Finally Contract the result of the
    Non Contracted operations with the Contracted Ones for getting
    the desired reduced density matrix. Try to alternate MPS contraction
    of states (free legs) with MPO contractions (contracted legs)

    Parameters
    ----------
    mps : list
        MPS representation of the state whose reduced density matrix
        want to be computed. Each element is a rank-3 tensor represented
        by a numpy array
    free_indices : list
        List of the free index for computing the reduced density matrix
    contraction_indices : list
        List with the contraction indices for computing the reduced
        density matrix. This qubits will be traced out.

    Returns
    _______

    tensor_out : np.array
        Array with the desired reduced density matrix
    """
    # First deal with contraction indices
    logger.debug("MPS: %s", [a.shape for a in mps])
    tensor_contracted = contraction_physical_legs(mps[contraction_indices[0]])
    for i in contraction_indices[1:]:
        #print(i)
        tensor = contraction_physical_legs(mps[i])
        tensor_contracted = mpo_contraction(tensor_contracted, tensor)
    # tensor_contracted is a matrix formed with the tensors with
    # contracted physical legs

    first_dim = int(np.sqrt(tensor_contracted.shape[0]))
    second_dim = int(np.sqrt(tensor_contracted.shape[-1]))
    tensor_contracted = tensor_contracted.reshape([
        first_dim, first_dim, second_dim, second_dim])

    # Second deal with free indices
    tensor_free = mps[free_indices[0]]
    for i in free_indices[1:]:
        #print(i)
        tensor = mps[i]
        #print(tensor.shape)
        tensor_free = contract_indices(tensor_free, tensor, [2], [0])
        #print(tensor_free.shape)
        reshape = [
            tensor_free.shape[0],
            tensor_free.shape[1] * tensor_free.shape[2],
            tensor_free.shape[3],
        ]
        tensor_free = tensor_free.reshape(reshape)
    logger.debug("Free tensor: %s", tensor_free.shape)
    logger.debug("Contracted tensor: %s", tensor_contracted.shape)
    # tensor free is the result of the contractions of tensors with
    # non contracted physical legs
    tensor_out = contract_indices(
        tensor_free, tensor_contracted, [2, 0], [0, 2])
    logger.debug("Output tensor: %s", tensor_out.shape)
    tensor_out = contract_indices(
        tensor_out, tensor_free.conj(), [1, 2], [2, 0])
    logger.debug("Output tensor: %s", tensor_out.shape)

    return tensor_out

def get_null_projectors(array):
    """
    Given an input matrix this function computes the matrix of projectors
    over the null space of the input matrix

    Parameters
    ----------

    array : numpy array
        Input matrix for computing the projectors over the null space

    Returns
    _______

    h_null : numpy array
        Matrices with the projectors to the null space of the
        input array
    """

    if np.linalg.matrix_rank(array) == len(array):
        text = "PROBLEM! The rank of the matrix is equal to its dimension.\
            Input matrix DOES NOT HAVE null space"
        raise ValueError(text)

    # Compute the basis vector for the null space
    v_null = linalg.null_space(array)
    # Computing the projectors to the null space
    h_null = v_null @ np.conj(v_null.T)
    return h_null

def get_local_reduced_matrix(state, qb_pos):
    """
    Given a MPS representation of a input state and position qubit
    (qb_pos) computes the minimum local reduced density matrix from
    qb_pos qubit such that its null space can be computable:
    i.e. rank(local_rho) < dim(local_rho)

    Parameters
    ----------

    state : numpy array
        MPS representation of an state
    qb_pos : int
        position of the qubit for computing the reduced density matrix

    Returns
    _______

    local_qubits : list
        list with the qubits affected by the reduced density matrix
    local_rho : numpy array
        array with the reduced density matrix for the input qbit
        position
    """
    logger.debug("Computing local reduced density matrix: qb_pos: %d", qb_pos)
    nqubit = len(state)
    state_index = list(range(nqubit))
    if qb_pos not in state_index:
        text = "qb_pos: {} NOT IN: {}".format(qb_pos, state_index)
        raise ValueError(text)
    group_qbits = 1
    stop = True
    while stop:
        # Iteration for grouping one qubit more in each step of the loop
        free_indices = [(qb_pos + k) % nqubit \
            for k in range(group_qbits + 1)]
        logger.debug("\t free_indices: %s", free_indices)
        # The contraction indices are built
        #contraction_indices = [
        #    i for i in state_index if i not in free_indices
        #]
        contraction_indices = [(free_indices[-1] + 1 + i) % nqubit  \
            for i in state_index]
        contraction_indices = list(filter(
            lambda x: x not in free_indices, contraction_indices))

        logger.debug("\t contraction_indices: %s", contraction_indices)
        # Computing the reduced density matrix
        rho = reduced_rho_mps(state, free_indices, contraction_indices)
        # Computes the rank of the obtained reduced matrix
        rank = np.linalg.matrix_rank(rho)
        logger.debug("\t rank: %d. Dimension: %d", rank, len(rho))
        if rank < len(rho):
            # Now we can compute a null space for reduced density operator
            logger.debug("\t Grouped Qubits: %s", free_indices)
            # Store the local qubits for each qubit
            local_qubits = free_indices
            # Store the local reduced density matrices for each qubit
            local_rho = rho
            stop = False
        group_qbits = group_qbits + 1

        if group_qbits == nqubit:
            stop = False
            local_rho = rho
            local_qubits = free_indices
        logger.debug("\t STOP: %s", stop)
    return local_qubits, local_rho

class PH_MPS:
    """
    Class for doing Parent Hamiltonian Computations

    Parameters
    ----------

    amplitudes : list
        list with the complete amplitudes of the state of the ansatz

    kwars : dictionary
        dictionary that allows the configuration of the CQPEAE algorithm:
            Implemented keys:

    """

    def __init__(self, mps, t_invariant=False, **kwargs):
        """

        Method for initializing the class

        """

        # Setting attributes
        self.mps_state = mps
        self.nqubits = len(self.mps_state)
        self.t_invariant = t_invariant
        # For Saving
        self._save = kwargs.get("save", False)
        self.filename = kwargs.get("filename", None)
        # Float precision for removing Pauli coefficients
        self.float_precision = np.finfo(float).eps
        # For storing the reduced density matrix
        self.reduced_rho = None
        # For storing the non trace out qubits of the reduced
        # density matrix
        self.local_free_qubits = None
        # For storing the different comnputed local kernel projectors
        self.local_projectors = None
        self.qubits_list = None
        self.local_parent_hamiltonians = None

        self.rho = None
        self.naive_parent_hamiltonian = None

        self.pauli_coeficients = None
        self.pauli_strings = None
        self.pauli_pdf = None
        self.ph_time = None


    # def get_density_matrix(self):
    #     """
    #     Computes the density matrix asociated with the amplitudes
    #     """
    #     #Convert MPS to a Vector
    #     state = self.amplitudes.reshape((-1, 1))
    #     # Create density matrix
    #     self.rho = state @ np.conj(state.T)

    def get_pauli_pdf(self):
        """
        Create pauli dataframe with all the info.
        Additionally pauli coefficients lower than float precision will
        be pruned
        """
        values = [self.pauli_coeficients, self.pauli_strings, self.qubits_list]
        pauli_pdf = pd.DataFrame(
            values,
            index=['PauliCoefficients', 'PauliStrings', 'Qbits']).T
        len_coefs = len(pauli_pdf)
        self.pauli_pdf = pauli_pdf[
            abs(pauli_pdf["PauliCoefficients"]) > self.float_precision]
        logger.info(
            "Number Pauli Coefficients: %d. Number of prunned coefs: %d",
            len_coefs,
            len(self.pauli_pdf)
        )


    def local_ph(self):
        """
        Computes the local parent hamiltonian

        """

        logger.debug("Computing Local Parent Hamiltonian")
        tick = time.time()
        self.reduced_rho = []
        self.local_free_qubits = []
        self.local_parent_hamiltonians = []
        self.qubits_list = []
        self.pauli_coeficients = []
        self.pauli_strings = []

        if self.t_invariant:
            # For translational invariant ansatz only reduced density
            # matrix for one qubit should be computed
            print("Invariant Ansatz: Only First Qubit computations")
            iterator = [0]
        else:
            # Reduced density matrices for all aqubits should be computed
            iterator = range(self.nqubits)


        for qb_pos in iterator:
            # Computing local reduced density matrix of the qubit
            logger.info("Reduced density Matrix Computations. Start")
            lq, lrho = get_local_reduced_matrix(self.mps_state, qb_pos)
            logger.info("Reduced density Matrix Computations. End")
            self.local_free_qubits = self.local_free_qubits + [lq]
            self.reduced_rho = self.reduced_rho + [lrho]
            # Computing projector on null space for the qubit
            rho_null_projector = get_null_projectors(lrho)
            logger.debug("Computing Null Projectors: qb_pos: %s", qb_pos)
            self.local_parent_hamiltonians.append(rho_null_projector)
            logger.debug(
                "Computing Decomposition in Pauli Matrices: qb_pos: %s", qb_pos)
            # Decomposition in pauli matrices
            coefs, paulis = pauli_decomposition(rho_null_projector, len(lq))
            self.pauli_coeficients = self.pauli_coeficients + coefs
            self.pauli_strings = self.pauli_strings + paulis
            self.qubits_list = self.qubits_list + [lq for i in paulis]

        # if self.t_invariant:
        #     # For translational invariant ansatzes we must replicate
        #     # the pauli terms for all the qubits
        #     terms = len(self.pauli_coeficients)
        #     self.pauli_coeficients = self.pauli_coeficients * self.nqubits
        #     self.pauli_strings = self.pauli_strings * self.nqubits
        #     self.qubits_list = []
        #     for qb_pos in range(self.nqubits):
        #         step = [(qb_pos + k) % self.nqubits for k in range(
        #             len(self.local_free_qubits[0]))]
        #         self.qubits_list = self.qubits_list + [step] * terms

        self.get_pauli_pdf()
        tack = time.time()
        self.ph_time = tack - tick
        if self._save:
            self.save()

    #def naive_ph(self):
    #    """
    #    Computes the parent hamiltonian of agiven ansatz for a full qubit
    #    interaction

    #    """

    #    logger.debug("Computing Naive Parent Hamiltonian")
    #    tick = time.time()
    #    if self.nqubits > 11:
    #        text = "The number of elements of the linear combination \
    #        scales as 4^n. Decomposition can be only done for n <=11"
    #        raise ValueError(text)
    #    # Compute the density matrix
    #    logger.debug("Computing Density Matrix")
    #    self.get_density_matrix()
    #    # Compute Parent Hamiltonian
    #    logger.debug("Computing Projectors on Null space")
    #    self.naive_parent_hamiltonian = get_null_projectors(self.rho)
    #    logger.debug("Computing Decomposition in Pauli Matrices")
    #    self.pauli_coeficients, self.pauli_strings = pauli_decomposition(
    #        self.naive_parent_hamiltonian, self.nqubits)
    #    self.qubits_list = [list(range(self.nqubits))] \
    #        * len(self.pauli_coeficients)
    #    self.get_pauli_pdf()
    #    tack = time.time()
    #    self.ph_time = tack - tick

    def save(self):
        """
        Saving Staff
        """
        self.pauli_pdf.to_csv(
            self.filename+"_pauli.csv", sep=";")
        pdf = pd.DataFrame(
            [self.ph_time], index=["ph_time"]).T
        pdf.to_csv(self.filename+"_ph_time.csv", sep=";")

def run_parent_hamiltonian(**configuration):
    """
    Computes Parent Hamiltonian for an ansatz

    Parameters
    ----------
    configuration is a kwargs. Following keys are the used:
    * nqubits : int
        number of the qubits for the BTC ansatz
    * depth : int
        depth for the BTC ansatz
    * truncate : bool
        for truncating the SVD. Float precision will be used as cuttoff
    * save : bool
        for saving the results
    * folder : str
        Folder for storing the results.
    Returns
    -------

    pauli_pdf : pandas DataFrame
        DataFrame with the Pauli decomposition of hte PH
    pdf_angles : pandas DataFrame
        DataFrame with the angles for the BTC ansatz.
    """
    # Ansatz configuration
    nqubits = configuration["nqubits"]
    depth = configuration["depth"]
    truncate = configuration["truncate"]
    t_v = configuration["t_v"]
    # PH configuration
    #t_inv = configuration["t_inv"]
    #MPS only available for transaltional invariant ansatz
    t_inv = configuration["t_inv"]
    save = configuration["save"]
    folder_name = configuration["folder"]
    base_fn = ""
    if save:
        folder_name = create_folder(folder_name)
        base_fn = folder_name + \
            "ansatz_simple01_nqubits_{}_depth_{}_qpu_mps".format(
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
    # Configuring PH computations
    ph_conf = {"save": save, "filename":base_fn}
    # Computing Parent Hamniltonian using MPS
    ph_ob_mps = PH_MPS(mps, t_inv, **ph_conf)
    ph_ob_mps.local_ph()
    return ph_ob_mps.pauli_pdf, pdf_angles

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s-%(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        #level=logging.INFO
        level=logging.DEBUG
    )
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
        "-t_v",
        dest="t_v",
        type=float,
        help="Truncation Value for SVD in the MPS computations",
        default=None,
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
        pdf, angles = run_parent_hamiltonian(**vars(args))
        print(pdf)


    # folder = "/mnt/netapp1/Store_CESGA/home/cesga/gferro/PH/"
    # folder_n = "ansatz_simple01_nqubits_20_depth_3_qpu_ansatz_mps/"
    # # Read the state from csv
    # base_fn = get_filelist(folder + folder_n)[0]
    # logger.info("Loading State")
    # ph_time = ph_object.ph_time
    # logger.info("Computed Local Parent Hamiltonian in: %s", ph_time)
    # pdf_ph_time = pd.DataFrame([ph_time], index=["ph_time"]).T
    # if args.save:
    #     pdf_ph_time.to_csv(base_fn+"_ph_time.csv", sep=";")

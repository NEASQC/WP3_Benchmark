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
from pauli import pauli_decomposition
from contractions import reduced_matrix
import logging
logger = logging.getLogger('__name__')


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
    nqubit = state.ndim
    state_index = list(range(nqubit))
    if qb_pos not in state_index:
        text = "qb_pos: {} NOT IN: {}".format(qb_pos, state_index)
        raise ValueError(text)
    group_qbits = 1
    stop = True
    while stop:
        # Iteration for grouping one qubit more in each step of the loop
        free_indices = [(qb_pos + k)%nqubit for k in range(group_qbits + 1)]
        logger.debug("\t free_indices: %s", free_indices)
        # The contraction indices are built
        contraction_indices = [
            i for i in state_index if i not in free_indices
        ]
        logger.debug("\t contraction_indices: %s", contraction_indices)
        # Computing the reduced density matrix
        rho = reduced_matrix(state, free_indices, contraction_indices)
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

class PH:
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

    def __init__(self, amplitudes, t_invariant=False, **kwargs):
        """

        Method for initializing the class

        """

        # Setting attributes
        self.amplitudes = np.array(amplitudes)
        nqubits = float(np.log2(len(amplitudes)))
        if nqubits.is_integer() is not True:
            text = "The len of the amplitudes MUST BE 2^n"
            raise ValueError(text)
        self.nqubits = int(nqubits)
        self.t_invariant = t_invariant
        self.mps_state = self.amplitudes.reshape(
            tuple(2 for i in range(self.nqubits))
        )
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


    def get_density_matrix(self):
        """
        Computes the density matrix asociated with the amplitudes
        """
        #Convert MPS to a Vector
        state = self.amplitudes.reshape((-1, 1))
        # Create density matrix
        self.rho = state @ np.conj(state.T)

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
            lq, lrho = get_local_reduced_matrix(self.mps_state, qb_pos)
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

    def naive_ph(self):
        """
        Computes the parent hamiltonian of agiven ansatz for a full qubit
        interaction

        """

        logger.debug("Computing Naive Parent Hamiltonian")
        tick = time.time()
        if self.nqubits > 11:
            text = "The number of elements of the linear combination \
            scales as 4^n. Decomposition can be only done for n <=11"
            raise ValueError(text)
        # Compute the density matrix
        logger.debug("Computing Density Matrix")
        self.get_density_matrix()
        # Compute Parent Hamiltonian
        logger.debug("Computing Projectors on Null space")
        self.naive_parent_hamiltonian = get_null_projectors(self.rho)
        logger.debug("Computing Decomposition in Pauli Matrices")
        self.pauli_coeficients, self.pauli_strings = pauli_decomposition(
            self.naive_parent_hamiltonian, self.nqubits)
        self.qubits_list = [list(range(self.nqubits))] \
            * len(self.pauli_coeficients)
        self.get_pauli_pdf()
        tack = time.time()
        self.ph_time = tack - tick

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
    state_file = configuration["state"]
    base_fn = configuration["base_fn"]
    save = configuration["save"]
    t_inv = configuration["t_inv"]

    logger.info("Loading State")
    state = pd.read_csv(state_file, sep=";", index_col=0)
    print(state)

    # Create PH
    logger.info("Computing Local Parent Hamiltonian")
    amplitudes = state[["Amplitude"]].astype(complex)
    print(amplitudes)
    # Compute Local PH
    ph_conf = {
        "filename": base_fn,
        "save": save
    }
    ph_object = PH(amplitudes, t_inv, **ph_conf)
    ph_object.local_ph()

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s-%(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO
        #level=logging.DEBUG
    )
    logger = logging.getLogger('__name__')
    # Given a state Compute its Parent Hamiltonian
    import argparse

    parser = argparse.ArgumentParser()

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
    parser.add_argument(
        "-state",
        dest="state",
        type=str,
        default="",
        help="Filename of the state",
    )
    parser.add_argument(
        "-basefn",
        dest="base_fn",
        type=str,
        default="",
        help="Base Filename for Saving Pauli Decomposition",
    )
    args = parser.parse_args()
    run_parent_hamiltonian(**vars(args))


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

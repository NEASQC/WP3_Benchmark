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
import sys
import numpy as np
from scipy import linalg
sys.path.append("../")
from PH.pauli import pauli_decomposition
from PH.contractions import reduced_matrix
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

def get_local_reduced_matrices(state):
    """
    Given a MPS representation of a input state computes the local
    reduced density matrices for each qubit of input state.

    Parameters
    ----------

    state : numpy array
        MPS representation of an state
    """
    #Getting the number of qubits of the MPS
    nqubit = state.ndim
    # Indexing for input MPS state
    state_index = list(range(nqubit))
    logger.info('state_index: {}'.format(state_index))
    local_qubits = []
    local_rho = []
    for qb_pos in range(nqubit):
        # Iteration over all the qubits positions
        logger.info('qb_pos: {}'.format(qb_pos))
        group_qbits = 1
        stop = False
        while stop == False:
            # Iteration for grouping one qubit more in each step of the loop
            free_indices = [(qb_pos + k)%nqubit for k in range(group_qbits + 1)]
            logger.debug('free_indices: {}'.format(free_indices))
            # The contraction indices are built
            contraction_indices = [
                i for i in state_index if i not in free_indices
            ]
            logger.debug('contraction_indices: {}'.format(contraction_indices))
            # Computing the reduced density matrix
            rho = reduced_matrix(
                state, free_indices, contraction_indices)
            # Computes the rank of the obtained reduced matrix
            rank = np.linalg.matrix_rank(rho)
            logger.debug('rank: {}. Dimension: {}'.format(rank, len(rho)))
            if rank < len(rho):
                # Now we can compute a null space for reduced density operator
                logger.debug('Grouped Qubits: {}'.format(free_indices))
                # Store the local qubits for each qubit
                local_qubits.append(free_indices)
                # Store the local reduced density matrices for each qubit
                local_rho.append(rho)
                stop = True
            group_qbits = group_qbits + 1

            if group_qbits == nqubit:
                stop = True
            logger.debug('STOP: {}'.format(stop))
    return local_qubits, local_rho


def parent_hamiltonian(state):
    """
    Given a MPS representation of a input state computes the local
    Hamiltonian terms in Pauli Basis. This terms allows to build
    the parent Hamiltonian of the input state.

    Parameters
    ----------

    state : numpy array
        MPS representation of an state
    Returns
    _______

    local_qubits : list
       list with the local qubits for each qubit of the initial state
    hamiltonian_coeficients : list
        list of the coefficients for each  Hamiltonian term
    hamiltonian_paulis : list
        list of Pauli strings for each Hamiltonian term.
    hamiltonian_local_qubits : list
        list with the qubits where the Hamiltonian term is applied
    """

    #First we compute the local reduced density matrices for each qubit
    local_qubits, local_rho = get_local_reduced_matrices(state)

    hamiltonian_coeficients = []
    hamiltonian_paulis = []
    hamiltonian_local_qubits = []
    for rho_step, free_indices in zip(local_rho, local_qubits):
        # Get the null projectors
        rho_null_projector = get_null_projectors(rho_step)
        # Compute Pauli decomposition of null projectors
        coefs, paulis = pauli_decomposition(
            rho_null_projector, len(free_indices)
        )
        hamiltonian_coeficients = hamiltonian_coeficients + coefs
        hamiltonian_paulis = hamiltonian_paulis + paulis
        hamiltonian_local_qubits = hamiltonian_local_qubits \
            + [free_indices for i in paulis]
    return hamiltonian_coeficients, hamiltonian_paulis, hamiltonian_local_qubits

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

    def __init__(self, amplitudes, **kwargs):
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
        self.mps_state = self.amplitudes.reshape(
            tuple(2 for i in range(self.nqubits))
        )
        # For storing the reduced density matrix
        self.reduced_rho = None
        # For storing the non trace out qubits of the reduced
        # density matrix
        self.local_free_qubits = None
        # For sotring the different comnputed local kernel projectors
        self.local_projectors = None
        self.qubits_list = None
        self.local_parent_hamiltonians = None

        self.rho = None
        self.naive_parent_hamiltonian = None

        self.pauli_coeficients = None
        self.pauli_matrices = None

    def get_reduced_density_matrices(self):
        """
        Method for computing the reduced density matrix for all the qubits
        """
        self.local_free_qubits, self.reduced_rho = get_local_reduced_matrices(
            self.mps_state)

    def get_density_matrix(self):
        """
        Computes the density matrix asociated with the amplitudes
        """
        #Convert MPS to a Vector
        state = self.amplitudes.reshape((-1, 1))
        # Create density matrix
        self.rho = state @ np.conj(state.T)

    def get_parent_hamiltonian(self, rho):
        """
        Method for computing the null projectors of a matrix
        """
        rho_null_projector = get_null_projectors(rho)
        # Compute Pauli decomposition of null projectors
        return rho_null_projector

    def pauli_decomposition(self, array, dimension):
        """
        Creates the Pauli Decomposition of an input matrix

        Parameters
        ----------

        array : numpy array
            input array for doing the Pauli decomposition
        dimension : int
            dimension for creating the corresponding basis of Kronecker
            products of Pauli basis

        Returns
        -------

        coefs : list
            Coefficients of the different Pauli matrices decomposition
        strings_pauli : list
            list with the complete Pauli string decomposition

        """
        if dimension > 11:
            text = "The number of elements of the linear combination \
            scales as 4^n. Decomposition can be only done for n <=11"
            raise ValueError(text)
        coefs, paulis = pauli_decomposition(array, dimension)
        return coefs, paulis

    def local_ph(self):
        """
        Computes the local parent hamiltonian

        """

        self.get_reduced_density_matrices()
        self.local_parent_hamiltonians = []

        self.qubits_list = []
        self.pauli_coeficients = []
        self.pauli_matrices = []

        iterator = zip(self.reduced_rho, self.local_free_qubits)
        for rho_step, free_indices in iterator:
            # Get the null projectors
            rho_null_projector = self.get_parent_hamiltonian(rho_step)
            self.local_parent_hamiltonians.append(rho_null_projector)
            # Compute Pauli decomposition of null projectors
            coefs, paulis = self.pauli_decomposition(
                rho_null_projector, len(free_indices)
            )
            self.pauli_coeficients = self.pauli_coeficients + coefs
            self.pauli_matrices = self.pauli_matrices + paulis
            self.qubits_list = self.qubits_list + [free_indices for i in paulis]

    def naive_ph(self):
        """
        Computes the parent hamiltonian of agiven ansatz for a full qubit
        interaction

        """

        # Compute the density matrix
        self.get_density_matrix()
        # Compute Parent Hamiltonian
        self.naive_parent_hamiltonian = self.get_parent_hamiltonian(self.rho)
        self.pauli_coeficients, self.pauli_matrices = pauli_decomposition(
            self.naive_parent_hamiltonian, self.nqubits)

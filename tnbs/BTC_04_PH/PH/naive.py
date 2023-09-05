"""
This package contains all the functions for a naive implementation
of the parent hamiltonian
"""

import logging
import numpy as np
from scipy import linalg
from parent_hamiltonian.pauli.pauli_decomposition import pauli_decomposition
logger = logging.getLogger('__name__')


def parent_hamiltonian(ansatz):
    """
    Computes the parent hamiltonian of agiven ansatz for a full qubit
    interaction

    Parameters
    ----------

    ansatz : numpy array
        MPS representation of the state of the ansatz


    Returns
    _______

    coefs : list
        List of coefficients of each term of the parent hamiltonian
    pauli_strings : list
        List of the pauli string of each term of the parent hamiltonian
    rho : numpy array
        Density operator of the input ansatz
    kernel : numpy array
        Basis of the null space of the density operator of the ansatz.
    null_projector_ : numpy array
        Projector matrix to the null space of the density operator of
        the ansatz
    """

    #Convert MPS to a Vector
    state = ansatz.reshape((2 ** ansatz.ndim, 1))
    # Create density matrix
    rho = state @ np.conj(state.T)
    # Compute kernel
    kernel = linalg.null_space(rho)
    #Compute projectors on null space
    null_projector_ = kernel @ np.conj(kernel.T)
    coefs, pauli_strings = pauli_decomposition(
        null_projector_, ansatz.ndim
    )
    return coefs, pauli_strings, rho, kernel, null_projector_

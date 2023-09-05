"""
This module contains all functions needed for creating the Pauli
decomposition of a Hamiltonian

Author: Gonzalo Ferro
"""

import itertools as it
import functools as ft
from joblib import Parallel, delayed
import numpy as np

def pauli_operators(pauli_index):
    """
    Return the correspondent Pauli matrix.

    Parameters
    ----------

    pauli_index : int
        Number for selecting the Pauli matrix:
        0 -> identity, 1-> X, 2-> Y, 3 -> Z

    Returns
    -------

    pauli = np.array
        2x2 Pauli Matrix

    """
    if pauli_index == 0:
        pauli = np.identity(2, dtype=np.complex128)
    elif pauli_index == 1:
        pauli = np.array([[0, 1], [1, 0]])
    elif pauli_index == 2:
        pauli = np.array([[0, -1j], [1j, 0]])
    elif pauli_index == 3:
        pauli = np.array([[1, 0], [0, -1]])
    return pauli


def pauli_strings(pauli_list):
    """
    Provides the Pauli strings of the input list.

    Parameters
    ----------

    pauli_list : list
        list with the ints of the Pauli matrices. See pauli_operators for
        the mapping

    Returns
    -------

    string_pauli : str
        correspondent Pauli strings of the input

    """
    pauli_string = ''.join([str(i) for i in pauli_list])
    string_pauli = pauli_string.replace("0", "I").replace("1", "X")\
        .replace("2", "Y").replace("3", "Z")
    return string_pauli


def pauli_inner_product(array, pauli_list, dimension):
    """
    Given an input matrix and a Kronecker Pauli matrix product computes
    its inner product. Under the hood it computes the Frobenius norm
    of the complex conjugate of the matrix by the given Kronecker Pauli
    matrix product.

    Parameters
    ----------

    array : numpy array
        Input array for getting the inner product
    pauli_list : list
        list with the indices to compute the Kronecker product of the
        corresponding Pauli matrices.Each element of the list must be
        a number between 0 and 3 corresponding to the Pauli matrix
        following convention from pauli_operators function
    dimension : int
        For doing the normalisation corresponding to the dimension of
        the matrices

    Returns
    -------

    inner_product : float
        Resulting inner product

    """
    # The matrix corresponding to the input Kronecker product of Pauli
    # matrices is computed
    step_pauli = ft.reduce(
        np.kron, [pauli_operators(i) for i in pauli_list])
    # The inner product between the input array and the desired
    # Kronecker product of the Pauli matrices is computed
    inner_product = np.einsum('ij,ij->', np.conj(array), step_pauli).real \
        / 2 ** dimension
    return inner_product

def pauli_decomposition(array, dimension, jobs=-1):
    """

    Creates the Pauli Decomposition of an input matrix

    Parameters
    ----------

    array : numpy array
        input array for doing the Pauli decomposition
    dimension : int
        dimension for creating the corresponding basis of Kronecker
        products of Pauli basis
    jobs = int
        For speed up the coefficients computations. Number of parallel
        computations.

    Returns
    -------

    coefs : list
        Coefficients of the different Pauli matrices decomposition
    strings_pauli : list
        list with the complete Pauli string decomposition

    """

    if array.shape[0] != array.shape[1]:
        message = "BE AWARE. Matrix should have same dimensions"
        raise ValueError(message)


    strings_pauli = []
    coefs = []
    # Here we compute all possible Pauli combinations
    combinations = it.product([0, 1, 2, 3], repeat=dimension)
    # Correspondent Pauli strings
    strings_pauli = [pauli_strings(list(comb)) for comb in combinations]
    # Computation of the inner product between input array and all
    # possible Kronecker product of Pauli matrices. This here we compute
    # the coefficients of the array in the basis of the correspondent
    # dimension Kronecker product Pauli matrices.
    # This part can be very demanding so we try to parallelize as much
    # as we can.
    combinations = it.product([0, 1, 2, 3], repeat=dimension)
    coefs = Parallel(n_jobs=jobs)(
        delayed(pauli_inner_product)(
            array, comb, dimension
        ) for comb in combinations
    )
    return coefs, strings_pauli

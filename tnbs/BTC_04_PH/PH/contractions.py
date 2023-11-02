"""
Implementation of reduced density matrix computations and
tensor contractions

Author: Gonzalo Ferro
"""

import string
import logging
import re
import numpy as np
logger = logging.getLogger('__name__')


def contract_indices_old(tensor1, tensor2, contraction1, contraction2):
    """
    Compute the contraction of 2 input tensors for the input contraction
    indices.The computation is done by, transposing indices, reshaping
    and doing matrix multiplication

    Parameters
    ----------

    tensor1 : numpy array
        first tensor
    tensor2 : numpy array
        second tensor
    contraction1 : list
        contraction indices for first tensor
    contraction2 : list
        contraction indices for second tensor

    Returns
    _______

    rho : numpy array
        Desired reduced density operator in matrix form

    """

    if len(contraction1) != len(contraction2):
        raise ValueError("Different number of contraction indices!")
    indices1 = list(range(tensor1.ndim))
    indices2 = list(range(tensor2.ndim))

    # Free indices for tensors
    free_indices1 = [i for i in indices1 if i not in contraction1]
    free_indices2 = [i for i in indices2 if i not in contraction2]

    # Transpose elements
    tensor1_t = tensor1.transpose(free_indices1 + contraction1)
    tensor2_t = tensor2.transpose(free_indices2 + contraction2)

    tensor1_t_matrix = tensor1_t.reshape(
        len(tensor1) ** len(free_indices1),
        len(tensor1) ** len(contraction1)
    )
    tensor2_t_matrix = tensor2_t.reshape(
        len(tensor2) ** len(free_indices2),
        len(tensor2) ** len(contraction2)
    )
    contraction_tensor = tensor1_t_matrix @ tensor2_t_matrix.T
    return contraction_tensor

def contract_indices(tensor1, tensor2, contraction1=[], contraction2=[]):
    """
    Compute the contraction of 2 input tensors for the input contraction
    indices.The computation is done by, transposing indices, reshapin
    and doing matrix multiplication. Tensor legs can be of different
    dimension.
    BE AWARE: Order in contraction indices it is very important:
    contraction1 = [1, 2, 5] contraction2 = [2, 0, 6] -> Contractions
    will be: [1-2, 2-0, 5-6]
    But if contraction2 = [0, 2, 6] then contraction
    will be: [1-0, 2-2,5-6]
    If both contraction indices are empty then the tensors will be glued

    Parameters
    ----------

    tensor1 : numpy array
        first tensor
    tensor2 : numpy array
        second tensor
    contraction1 : list
        contraction indices for first tensor
    contraction2 : list
        contraction indices for second tensor

    Returns
    _______

    rho : numpy array
        Desired reduced density operator in matrix form

    """

    if len(contraction1) != len(contraction2):
        raise ValueError("Different number of contraction indices!")
    indices1 = list(range(tensor1.ndim))
    indices2 = list(range(tensor2.ndim))

    # Free indices for tensors
    free_indices1 = [i for i in indices1 if i not in contraction1]
    free_indices2 = [i for i in indices2 if i not in contraction2]

    # Transpose elements
    tensor1_t = tensor1.transpose(free_indices1 + contraction1)
    tensor2_t = tensor2.transpose(free_indices2 + contraction2)

    # If free_indices are empty
    if len(free_indices1) == 0:
        free_indices_1 = 1
    else:
        free_indices_1 = np.prod(
            [tensor1.shape[i] for i in free_indices1])

    # If contraction1 is empty
    if len(contraction1) == 0:
        contraction_indices_1 = 1
    else:
        contraction_indices_1 = np.prod(
            [tensor1.shape[i] for i in contraction1])

    # tensor1_t_matrix = tensor1_t.reshape(
    #     free_indices_1,
    #     np.prod([tensor1.shape[i] for i in contraction1]),
    # )

    tensor1_t_matrix = tensor1_t.reshape(
        free_indices_1, contraction_indices_1)

    # If free_indices are empty
    if len(free_indices2) == 0:
        free_indices_2 = 1
    else:
        free_indices_2 = np.prod(
            [tensor2.shape[i] for i in free_indices2])

    # If contraction1 is empty
    if len(contraction2) == 0:
        contraction_indices_2 = 1
    else:
        contraction_indices_2 = np.prod(
            [tensor2.shape[i] for i in contraction2])

    # tensor2_t_matrix = tensor2_t.reshape(
    #     free_indices_2,
    #     np.prod([tensor2.shape[i] for i in contraction2]),
    # )

    tensor2_t_matrix = tensor2_t.reshape(
        free_indices_2, contraction_indices_2)

    # Do the bare matrix multiplication
    contraction_tensor = tensor1_t_matrix @ tensor2_t_matrix.T

    free_contraction = [tensor1.shape[i] for i in free_indices1] + \
        [tensor2.shape[i] for i in free_indices2]
    contraction_tensor = contraction_tensor.reshape(free_contraction)

    return contraction_tensor


def reduced_matrix(state, free_indices, contraction_indices):
    """
    Compute the reduced density matrix for the input contraction indices
    The computation is done by, transposing indices, reshaping and doing
    matrix multiplication

    Parameters
    ----------

    state : numpy array
        array in MPS format of the state for computing reduced density
        matrix
    free_indices : list
        Free indices of the MPS state (this is the qubit that will NOT
        be traced out)
    contraction_indices : list
        Indices of the MPS state that will be contracted to compute
        the correspondent reduced density matrix

    Returns
    _______

    rho : numpy array
        Desired reduced density operator in matrix form

    """
    if len(contraction_indices) + len(free_indices) != state.ndim:
        raise ValueError(
            "Dimensions of free_indices and contraction_indices not compatible\
            with dimension of state")
    # First Transpose indices
    transpose_state = state.transpose(free_indices+contraction_indices)
    # Second rearrange as a matrix
    matrix_state = transpose_state.reshape(
        len(state) ** len(free_indices), len(state) ** len(contraction_indices)
    )
    logger.debug('matrix_state_shape: {}'.format(matrix_state.shape))
    # Third Matrix Multiplication
    rho = matrix_state @ np.conj(matrix_state).T
    return rho

def reduced_matrix_string(state, free_indices, contraction_indices):
    """
    Compute the reduced density matrix for the input contraction indices
    The computation is done by using np.einsum. Limited to 52 indices.

    Parameters
    ----------

    state : numpy array
        array in MPS format of the state for computing reduced density
        matrix
    free_indices : list
        Free indices of the MPS state (this is the qubit that will NOT
        be traced out)
    contraction_indices : list
        Indices of the MPS state that will be contracted to compute
        the correspondent reduced density matrix

    Returns
    _______

    matriz_rho : numpy array
        Desired reduced density operator in matrix form

    """
    if len(contraction_indices) + len(free_indices) != state.ndim:
        raise ValueError(
            "Dimensions of free_indices and contraction_indices not compatible\
            with dimension of state")
    abc = string.ascii_uppercase
    # Indices for MPS
    mps_index = abc[:state.ndim]
    # Indices for MPS conjugated
    conj_mps_index = list(abc[:state.ndim])
    logger.debug('mps_index: {}'.format(mps_index))
    # The free indices in MPS conjugated will be lowercase
    for i in free_indices:
        conj_mps_index[i] = conj_mps_index[i].lower()
    conj_mps_index = ''.join(conj_mps_index)
    logger.debug('conj_mps_index: {}'.format(conj_mps_index))
    # Creation of form of the indices of the final tensor
    final_index = [mps_index[i] for i in free_indices]
    final_index = final_index + [conj_mps_index[i] for i in free_indices]
    final_index = ''.join(final_index)
    # String for Einstein summation convention
    einstein = '{0}, {1} -> {2}'.format(
        mps_index, conj_mps_index, final_index)
    logger.debug('Sum: {}'.format(einstein))
    # Computation of the reduced density operator
    rho = np.einsum(einstein, state, np.conj(state))
    dimension = len(re.search(r'[a-z]+', final_index)[0])
    matriz_rho = rho.reshape(2 ** dimension, 2 ** dimension)
    return matriz_rho

def contract_indices_string(tensor1, tensor2, contraction_indices):
    """
    Compute the contraction of 2 input tensors for the input contraction
    indices.The computation is done by, transposing indices, reshaping
    and doing matrix multiplication

    Parameters
    ----------

    tensor1 : numpy array
        array in MPS format of the state for computing reduced density
        matrix
    tensor2 : numpy array
        array in MPS format of the state for computing reduced density
        matrix
    contraction_indices : list
        Indices of the MPS state that will be contracted to compute
        the correspondent reduced density matrix

    Returns
    _______

    rho : numpy array
        Desired reduced density operator in matrix form

    """

    indices1 = list(range(tensor1.ndim))
    indices2 = list(range(tensor2.ndim))

    # Free indices for tensors
    free_indices1 = [i for i in indices1 if i not in contraction_indices]
    free_indices2 = [i for i in indices2 if i not in contraction_indices]

    # Transpose elements
    tensor1_t = tensor1.transpose(free_indices1+contraction_indices)
    tensor2_t = tensor2.transpose(free_indices2+contraction_indices)
    tensor1_t_matrix = tensor1_t.reshape(
        len(tensor1) ** len(free_indices1),
        len(tensor1) ** len(contraction_indices)
    )
    tensor2_t_matrix = tensor2_t.reshape(
        len(tensor2) ** len(free_indices2),
        len(tensor2) ** len(contraction_indices)
    )
    contraction_tensor = tensor1_t_matrix @ tensor2_t_matrix.T
    return contraction_tensor

"""
Functions for working with MPS
"""
import sys
import logging
import numpy as np
from scipy.linalg import svd
sys.path.append("../../")
from PH.parent_hamiltonian.contractions import contract_indices
logger = logging.getLogger('__name__')

def my_svd(array, truncate=False, t_v=None):
    """
    Execute SVD.

    Parameters
    ----------
    array : np.array

        Arry with matrix for SVD
    truncate : Bool
        For truncating SVD. If t_v is None then the truncation will be
        using float prcision of the system
    t_v : float
        In truncate is True then the t_v float will use as truncation
        threshold

    Returns
    -------

    u_, s, vh : np.arrays
        numpy arrays with the SVD of the input array
    """
    u_, s_, vh = svd(array, full_matrices=False)
    logger.debug("Before Truncation u_: %s", u_.shape)
    logger.debug("Before Truncation vh: %s", vh.shape)
    logger.debug("Before Truncation s_: %s", s_.shape)
    if truncate == True:
        # For truncate SVD
        logger.debug("Truncation s_: %s", s_)
        if t_v is None:
            # If not Truncation limit we use minimum float precision
            eps = np.finfo(float).eps
            u_ = u_[:, s_ > eps]
            vh = vh[s_ > eps, :]
            s_ = s_[s_ > eps]
        else:
            u_ = u_[:, s_ > t_v]
            vh = vh[s_ > t_v, :]
            s_ = s_[s_ > t_v]
        logger.debug("After Truncation u_: %s", u_.shape)
        logger.debug("After Truncation vh: %s", vh.shape)
        logger.debug("After Truncation s_: %s", s_.shape)
    return u_, s_, vh

def apply_local_gate(mps, gates):
    """
    Apply  local gates on several rank 3-tensors. For not apply a
    gate over a tensor a None should be provided.
    The rank-3 tensor MUST HAVE following indexing:

                        0-o-2
                          |
                          1

    Where 1 is the physical leg.
    The computation done is:
                    -o- -o- ... -o-
                     |   |       |
                     o   o       o
                     |   |       |
    Parameters
    ----------

    mps : list
        Each element is a rank-3 tensor from a MPS
    gates : list
        Each element is a local gate

    Returns
    _______

    o_qubits : list
        Each element is the resulting rank-3 tensor
    """
    o_qubits = []
    for q_, g_ in zip(mps, gates):
        if q_ is None:
            o_qubits.append(q_)
        else:
            o_qubits.append(contract_indices(
                q_, g_, [1], [0]).transpose(0, 2, 1))
    return o_qubits

def apply_2qubit_gate(tensor1, tensor2, gate=None, truncate=False, t_v=None):
    """
    Executes a 2-qubit gate between 2 rank-3 tensors
    The rank-3 tensors MUST HAVE following indexing:

                        0-o-2
                          |
                          1

    Where 1 is the physical leg.
    Following Computation is done:

        -o-o-
         | |  -> -o- -> SVD -> -o- -o-
         -o-                    |   |

    """
    logger.debug("tensor1: %s", tensor1.shape)
    logger.debug("tensor2: %s", tensor2.shape)
    d_physical_leg = tensor1.shape[1]

    if gate is None:
        return tensor1, tensor2
    else:
        # 0-o-o-3
        #   | |
        #   1 2
        step = contract_indices(tensor1, tensor2, [2], [0])
        # Reshape 2 physical legs to 1
        # 0-o-o-3     -o-
        #   | |   ->   |
        #   1 2      d1*d2
        step = step.reshape((tensor1.shape[0], -1, tensor2.shape[2]))
        logger.debug("Tensor1-Tensor2: %s", step.shape)
        # Contraction Tensor1-2 with Gate
        # 0-o-2    0-o-1    0-o-2
        #  1|0  ->   |   ->   |
        #   o        2        1
        #   |1
        step = contract_indices(step, gate, [1], [0])
        step = step.transpose(0, 2, 1)
        logger.debug("Tensor1-Gate-Tensor2: %s", step.shape)

        # SVD
        # Preparing SVD
        # -o- -> dim1 -o- dim2
        #  |
        dim1 = tensor1.shape[0] * tensor1.shape[1]
        dim2 = tensor2.shape[1] * tensor2.shape[2]
        step = step.reshape(dim1, dim2)

        logger.debug("Matrix for SVD : %s", step.shape)
        u_, s_, vh = my_svd(step, truncate, t_v)
        logger.debug("u_: %s", u_.shape)
        logger.debug("s_: %s", s_.shape)
        logger.debug("vh: %s", vh.shape)

        # Obtaining 2 new rank-3 tensors
        #dim1 -o- dim2 -> -o- -o-
        #                  |   |

        # Reshaping u_ as left tensor
        new_shape = [0, 0, 0]
        new_shape[1] = d_physical_leg
        new_shape[0] = tensor1.shape[0]
        new_shape[2] = -1
        #left = u_ @ np.diag(s_)
        left = u_
        left = left.reshape(tuple(new_shape))
        logger.debug("Left Tensor:, %s", left.shape)

        #logger.debug("Right Tensor:, %s", right.shape)
        new_shape = [0, 0, 0]
        new_shape[1] = d_physical_leg
        new_shape[0] = -1
        new_shape[2] = tensor2.shape[2]
        # Reshaping other part as right tensor
        right = np.diag(s_) @ vh
        # right = vh
        right = right.reshape(tuple(new_shape))
        logger.debug("Right Tensor:, %s", right.shape)

        return left, right

def apply_2qubit_gates(qubits, gates, truncate=True, t_v=1.0e-8):
    """
    Apply a circular ladder of Non-Local gates.

    Parameters
    ----------
    qubits : list
        MPS representation of a circuit. List where each element is a
        numpy array
    gates : list
        list where each element is a matrix representation of non-local
        gate
    truncate : bool
        Boolean variable for truncate SVD of the 2 qubits ansatz. If
        truncate is True and t_v is None precision of the float will be
        used as truncation value
    t_v: float
        Float for tuncating the SVD of the 2 qubits gates. Only valid if
        truncate is True,

    Returns
    _______
    new_qubits : list
        MPS representation of the resulting circuit of applied the
        circular ladder of non local gates over the input MPS. Each
        element of the list is a numpy array
    """
    new_qubits = [0 for i in qubits]
    left = qubits[0]
    for i in range(1, len(qubits)):
        right = qubits[i]
        gate = gates[i-1]
        #new_qubits[i-1], left = phase_change(left, right, gate)
        new_qubits[i-1], left = apply_2qubit_gate(
            left, right, gate, truncate=truncate, t_v=t_v)

    new_qubits[-1], new_qubits[0] = apply_2qubit_gate(
        left, new_qubits[0], gates[-1], truncate=truncate, t_v=t_v)
     #new_qubits[-1], new_qubits[0] = phase_change(left, new_qubits[0], gates[-1])
    return new_qubits

def compose_mps(mps):
    """
    Given an input MPS it computes the final Tensor

    Parameters
    ----------

    mps : list
        list where each element is a rank-3 tensor that conforms de MPS

    Returns
    _______

    tensor : np.array
        numpy array with the correspondient tensor

    """
    tensor = mps[0]
    for i in range(1, len(mps)):
        tensor = contract_indices(
            tensor, mps[i], [tensor.ndim-1], [0])
    return tensor

def contract_indices_one_tensor(tensor, contractions):
    """
    For an input tensor executes indices contractions by pair.
    The computation is done by contracting each para of indices
    with its corresponding identity matrix.
    EXAMPLE:
    contractions = [(1, 5), (3, 4)] Index 1 will be contracted with
    indice 5 and index 3 with index 4

    Parameters
    ----------

    tensor : numpy array
        input tensor
    contractions : list
        each element is a tuple of the indices to be contracted

    Returns
    _______

    tensor : numpy array
        Desired tensor with the corresponding contractions done

    """

    list_of_eyes = []
    indices = []
    for c_ in contractions:
        if tensor.shape[c_[0]] != tensor.shape[c_[1]]:
            raise ValueError("Problem with contraction: {}".format(c_))
        indices = indices + [c_[0], c_[1]]
        list_of_eyes.append(np.eye(tensor.shape[c_[0]]))
    tensor_out = list_of_eyes[0]
    for tensor_step in list_of_eyes[1:]:
        tensor_out = contract_indices(tensor_out, tensor_step)
    tensor_out = contract_indices(
        tensor, tensor_out, indices, list(range(tensor_out.ndim)))
    return tensor_out

def contraction_physical_legs(tensor):
    """
    Executes the contraction of the physical legs of the input rank-3
    tensor with its complex conjugate. Physical Legs MUST BE 1

    Parameters
    ----------
    tensor : np array
        Numpy array with a rank-3 tensor. Physical Legs MUST BE 1

    """
    tensor = contract_indices(tensor, tensor.conj(), [1], [1])
    tensor = tensor.transpose(0, 2, 1, 3)
    reshape = [
        tensor.shape[0] * tensor.shape[1],
        tensor.shape[2] * tensor.shape[3]
    ]
    tensor = tensor.reshape(reshape)
    return tensor

def mpo_contraction(tensor_1, tensor_2):
    """
    Contraction of 2 input tensors (npo tensors) with corresponding
    adjust of dimension for computing density matrices.
    The input tensors MUST be rank 4 or 2 tensors.

    Rank-4 tensor:    |  Rank-2 tensor:
         |            |
       - o -          |     - o -
         |            |

    Parameters
    ----------

    tensor_1 : np array
        First input 4 or 2 rank tensor
    tensor_2 : np array
        Second input 4 or 2 rank tensor

    Returns
    _______


    step : np array
        output rank 4 or 2 tensor
    """

    rank_tensor_1 = tensor_1.ndim
    rank_tensor_2 = tensor_2.ndim

    if (rank_tensor_1 == 4) and (rank_tensor_2 == 4):
        # Case 0
        tensor_out = contract_indices(tensor_1, tensor_2, [3], [0])
        tensor_out = tensor_out.transpose(0, 1, 3, 2, 4, 5)
        reshape = [
            tensor_out.shape[0],
            tensor_out.shape[1] * tensor_out.shape[2],
            tensor_out.shape[3] * tensor_out.shape[4],
            tensor_out.shape[5]
        ]
        tensor_out = tensor_out.reshape(reshape)
    elif (rank_tensor_1 == 4) and (rank_tensor_2 == 2):
        # Case 1
        tensor_out = contract_indices(tensor_1, tensor_2, [3], [0])
    elif (rank_tensor_1 == 2) and (rank_tensor_2 == 4):
        # Case 3
        tensor_out = contract_indices(tensor_1, tensor_2, [1], [0])
    elif (rank_tensor_1 == 2) and (rank_tensor_2 == 2):
        # Case 3
        tensor_out = contract_indices(tensor_1, tensor_2, [1], [0])
    else:
        raise ValueError("Input Tensors MUST be rank-4 or rank-2")
    return tensor_out


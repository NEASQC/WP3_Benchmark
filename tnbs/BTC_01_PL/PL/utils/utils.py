"""

This module contains several auxiliary functions needed by other scripts
of the library

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro Costas

Fast Walsh-Hadamard Transform is based on mex function written
by Chengbo Li@Rice Uni for his TVAL3 algorithm:

https://github.com/dingluo/fwht/blob/master/FWHT.py
"""

import numpy as np


def bitfield(n_int: int, size: int):
    """Transforms an int n_int to the corresponding bitfield of size size

    Parameters
    ----------
    n_int : int
        integer from which we want to obtain the bitfield
    size : int
        size of the bitfield

    Returns
    ----------
    full : list of ints
        bitfield representation of n_int with size size

    """
    aux = [1 if digit == "1" else 0 for digit in bin(n_int)[2:]]
    right = np.array(aux)
    left = np.zeros(max(size - right.size, 0))
    full = np.concatenate((left, right))
    return full.astype(int)

def fwht_natural(array: np.array):
    """Fast Walsh-Hadamard Transform of array x in natural ordering
    The result is not normalised
    Parameters
    ----------
    array : numpy array

    Returns
    ----------
    fast_wh_transform : numpy array
        Fast Walsh Hadamard transform of array x.

    """
    fast_wh_transform = array.copy()
    h_ = 1
    while h_ < len(fast_wh_transform):
        for i in range(0, len(fast_wh_transform), h_ * 2):
            for j in range(i, i + h_):
                x_ = fast_wh_transform[j]
                y_ = fast_wh_transform[j + h_]
                fast_wh_transform[j] = x_ + y_
                fast_wh_transform[j + h_] = x_ - y_
        h_ *= 2
    return fast_wh_transform


def fwht_sequency(x_input: np.array):
    """Fast Walsh-Hadamard Transform of array x_input in sequence ordering
    The result is not normalised
    Based on mex function written by Chengbo Li@Rice Uni for his TVAL3
    algorithm.
    His code is according to the K.G. Beauchamp's book -- Applications
    of Walsh and Related Functions.
    Parameters
    ----------
    x_input : numpy array

    Returns
    ----------
    x_output : numpy array
        Fast Walsh Hadamard transform of array x_input.

    """
    n_ = x_input.size
    n_groups = int(n_ / 2)  # Number of Groups
    m_in_g = 2  # Number of Members in Each Group

    # First stage
    y_ = np.zeros((int(n_ / 2), 2))
    y_[:, 0] = x_input[0::2] + x_input[1::2]
    y_[:, 1] = x_input[0::2] - x_input[1::2]
    x_output = y_.copy()
    # Second and further stage
    for n_stage in range(2, int(np.log2(n_)) + 1):
        y_ = np.zeros((int(n_groups / 2), m_in_g * 2))
        y_[0 : int(n_groups / 2), 0 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 0:m_in_g:2] + x_output[1:n_groups:2, 0:m_in_g:2]
        )
        y_[0 : int(n_groups / 2), 1 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 0:m_in_g:2] - x_output[1:n_groups:2, 0:m_in_g:2]
        )
        y_[0 : int(n_groups / 2), 2 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 1:m_in_g:2] - x_output[1:n_groups:2, 1:m_in_g:2]
        )
        y_[0 : int(n_groups / 2), 3 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 1:m_in_g:2] + x_output[1:n_groups:2, 1:m_in_g:2]
        )
        x_output = y_.copy()
        n_groups = int(n_groups / 2)
        m_in_g = m_in_g * 2
    x_output = y_[0, :]
    return x_output


def fwht_dyadic(x_input: np.array):
    """Fast Walsh-Hadamard Transform of array x_input in dyadic ordering
    The result is not normalised
    Based on mex function written by Chengbo Li@Rice Uni for his TVAL3
    algorithm.
    His code is according to the K.G. Beauchamp's book -- Applications
    of Walsh and Related Functions.
    Parameters
    ----------
    array : numpy array

    Returns
    ----------
    x_output : numpy array
        Fast Walsh Hadamard transform of array x_input.

    """
    n_ = x_input.size
    n_groups = int(n_ / 2)  # Number of Groups
    m_in_g = 2  # Number of Members in Each Group

    # First stage
    y_ = np.zeros((int(n_ / 2), 2))
    y_[:, 0] = x_input[0::2] + x_input[1::2]
    y_[:, 1] = x_input[0::2] - x_input[1::2]
    x_output = y_.copy()
    # Second and further stage
    for n_stage in range(2, int(np.log2(n_)) + 1):
        y_ = np.zeros((int(n_groups / 2), m_in_g * 2))
        y_[0 : int(n_groups / 2), 0 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 0:m_in_g:2] + x_output[1:n_groups:2, 0:m_in_g:2]
        )
        y_[0 : int(n_groups / 2), 1 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 0:m_in_g:2] - x_output[1:n_groups:2, 0:m_in_g:2]
        )
        y_[0 : int(n_groups / 2), 2 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 1:m_in_g:2] + x_output[1:n_groups:2, 1:m_in_g:2]
        )
        y_[0 : int(n_groups / 2), 3 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 1:m_in_g:2] - x_output[1:n_groups:2, 1:m_in_g:2]
        )
        x_output = y_.copy()
        n_groups = int(n_groups / 2)
        m_in_g = m_in_g * 2
    x_output = y_[0, :]
    return x_output

def fwht(x_input: np.array, ordering: str = "sequency"):
    """Fast Walsh Hadamard transform of array x_input
    Works as a wrapper for the different orderings
    of the Walsh-Hadamard transforms.

    Parameters
    ----------
    x_input : numpy array
    ordering: string
        desired ordering of the transform

    Returns
    ----------
    y_output : numpy array
        Fast Walsh Hadamard transform of array x_input
        in the corresponding ordering
    """

    if ordering == "natural":
        y_output = fwht_natural(x_input)
    elif ordering == "dyadic":
        y_output = fwht_dyadic(x_input)
    else:
        y_output = fwht_sequency(x_input)
    return y_output


def left_conditional_probability(initial_bins, probability):
    """
    This function calculate f(i) according to the Lov Grover and Terry
    Rudolph 2008 papper:
    'Creating superposition that correspond to efficiently integrable
    probability distributions'
    http://arXiv.org/abs/quant-ph/0208112v1

    Given a discretized probability and an initial number of bins
    the function splits each initial region in 2 equally regions and
    calculates the conditional probabilities for x is located in the
    left part of the new regions when x is located in the region that
    contains the corresponding left region

    Parameters
    ----------

    initial_bins : int
        Number of initial bins for splitting the input probabilities
    probability : np.darray.
        Numpy array with the probabilities to be load.
        initial_bins <= len(Probability)

    Returns
    ----------

    left_cond_prob : np.darray
        conditional probabilities of the new initial_bins+1 splits
    """
    # Initial domain division
    domain_divisions = 2 ** (initial_bins)
    if domain_divisions >= len(probability):
        raise ValueError(
            "The number of Initial Regions (2**initial_bins)\
        must be lower than len(probability)"
        )
    # Original number of bins of the probability distribution
    nbins = len(probability)
    # Number of Original bins in each one of the bins of Initial
    # domain division
    bins_by_dd = nbins // domain_divisions
    # probability for x located in each one of the bins of Initial
    # domain division
    prob4dd = [
        np.sum(probability[j * bins_by_dd : j * bins_by_dd + bins_by_dd])
        for j in range(domain_divisions)
    ]
    # Each bin of Initial domain division is splatted in 2 equal parts
    bins4_left_dd = nbins // (2 ** (initial_bins + 1))
    # probability for x located in the left bin of the new splits
    left_probabilities = [
        np.sum(probability[j * bins_by_dd : j * bins_by_dd + bins4_left_dd])
        for j in range(domain_divisions)
    ]
    # Conditional probability of x located in the left bin when x is located
    # in the bin of the initial domain division that contains the split
    # Basically this is the f(j) function of the article with
    # j=0,1,2,...2^(i-1)-1 and i the number of qubits of the initial
    # domain division
    with np.errstate(divide="ignore", invalid="ignore"):
        left_cond_prob = np.array(left_probabilities) / np.array(prob4dd)
    left_cond_prob[np.isnan(left_cond_prob)] = 0
    return left_cond_prob

def expmod(n_input: int, base: int):
    r"""For a pair of integer numbers, performs the decomposition:

    .. math::
        n_input = base^power+remainder

    Parameters
    ----------
    n_input : int
        number to decompose
    base : int
        basis

    Returns
    -------
    power : int
        power
    remainder : int
        remainder
    """
    power = int(np.floor(np.log(n_input) / np.log(base)))
    remainder = int(n_input - base**power)
    return (power, remainder)

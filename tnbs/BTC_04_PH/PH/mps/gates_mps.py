"""
Numpy Implementation of Several Quantum Gates
"""
import numpy as np

def z_rotation(angle):
    """
    Implement a Z rotation numpy array.

    Parameters
    ----------

    angle: float
        desired angle for the Z rotation

    Returns
    _______

    zr : numpy array
        Array with the implementation of a Z rotation

    """
    zr = np.array([
        [np.exp(-0.5j*angle), 0],
        [0, np.exp(0.5j*angle)]
    ])
    # original
    #zr = np.array([
    #    [np.exp(0.5j*angle), 0],
    #    [0, np.exp(-0.5j*angle)]
    #])
    return zr

def x_rotation(angle):
    """
    Implement a X rotation numpy array.

    Parameters
    ----------

    angle: float
        desired angle for the Z rotation

    Returns
    _______

    xr : numpy array
        Array with the implementation of a X rotation

    """
    xr = np.array([
        [np.cos(0.5*angle), -1j*np.sin(0.5*angle)],
        [-1j*np.sin(0.5*angle), np.cos(0.5*angle)]
    ])
    # Originally
    #xr = np.array([
    #    [np.cos(0.5*angle), 1j*np.sin(0.5*angle)],
    #    [1j*np.sin(0.5*angle), np.cos(0.5*angle)]
    #])
    return xr


def phasechange_gate():
    """
    Implement a Phase Change Gate as a numpy array.

    Returns
    _______

    phase : numpy array
        Array with the implementation of a Phase change gate

    """
    phase = np.array([
        [1, 1],
        [1, -1]
    ])
    return phase

def h_gate():
    """
    Implement a Haddamard Gate as a numpy array.

    Returns
    _______

    haddamard : numpy array
        Array with the implementation of a Haddamard gate

    """
    haddamard = np.array([[1, 1], [1, 1]]) / np.sqrt(2)
    return haddamard

def controlz():
    """
    Implement a Controlled-Z Gate as a numpy array.

    Returns
    _______

    cz : numpy array
        Array with the implementation of a c-Z gate

    """
    cz=np.eye(4)
    cz[3, 3] = -1
    return cz

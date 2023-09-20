"""
Utils for PH benchmark test case
Author: Gonzalo Ferro
"""
import os

def create_folder(folder_name):
    """
    Check if folder exist. If not the function creates it

    Parameters
    ----------

    folder_name : str
        Name of the folder

    Returns
    ----------

    folder_name : str
        Name of the folder
    """

    # Check if the folder already exists
    if not os.path.exists(folder_name):
        # If it does not exist, create the folder
        os.mkdir(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
    if folder_name.endswith('/') != True:
        folder_name = folder_name + "/"
    return folder_name


def get_qpu(qpu=None):
    """
    Function for selecting solver.

    Parameters
    ----------

    qpu : str
        * qlmass: for trying to use QLM as a Service connection
            to CESGA QLM
        * python: for using PyLinalg simulator.
        * c: for using CLinalg simulator
        * mps: for using mps

    Returns
    ----------

    linal_qpu : solver for quantum jobs
    """

    if qpu is None:
        raise ValueError(
            "qpu CAN NOT BE NONE. Please select one of the three" +
            " following options: qlmass, python, c")
    if qpu == "qlmass":
        try:
            from qlmaas.qpus import LinAlg
            linalg_qpu = LinAlg()
        except (ImportError, OSError) as exception:
            raise ImportError(
                "Problem Using QLMaaS. Please create config file" +
                "or use mylm solver") from exception
    elif qpu == "mps":
        try:
            from qlmaas.qpus import MPS
            linalg_qpu = MPS(lnnize=True)
        except (ImportError, OSError) as exception:
            raise ImportError(
                "Problem Using QLMaaS. Please create config file" +
                "or use mylm solver") from exception
    elif qpu == "python":
        from qat.qpus import PyLinalg
        linalg_qpu = PyLinalg()
    elif qpu == "c":
        from qat.qpus import CLinalg
        linalg_qpu = CLinalg()
    elif qpu == "qat_qlmass":
        from qat.qpus import LinAlg
        linalg_qpu = LinAlg()
    elif qpu == "qat_mps":
        from qat.qpus import MPS
        linalg_qpu = MPS(lnnize=True)
    else:
        raise ValueError(
            "Invalid value for qpu. Please select one of the three "+
            "following options: qlmass, python, c")
    #print("Following qpu will be used: {}".format(linalg_qpu))
    return linalg_qpu

"""
Utils for PH benchmark test case
Author: Gonzalo Ferro
"""
import os
import re
import itertools as it

def combination_for_dictionary(input_dict):
    """
    Creates a list of dictionaries with all the posible combination of
    the input dictionary.

    Parameters
    ----------
    input_dict : python dictionary
        python dictionary where each key value MUST be a list. For each
        value of a list a new dictioanry will be created

    Returns
    ----------
    list_of_dictionaries : list of python dictionaries
        A list with all posible combination of dictionaries from the
        input dictionary
    """

    list_of_dictionaries = [
        dict(zip(input_dict, x)) for x in it.product(*input_dict.values())
    ]
    return list_of_dictionaries

def combination_for_list(input_list):
    """
    For each dictionary of the list the function creates all posible
    combinations. All the posible combinations are concatenated.

    Parameters
    ----------
    input_list : list of python dictionary
        The values of each key of the each python dictionary MUST BE lists.

    Returns
    ----------
    combo_list : list of python dictionaries
        A list with  the concatenation of all posible combinations for
        each dictionary of the input_list
    """
    combo_list = []
    for step_dict in input_list:
        combo_list = combo_list + combination_for_dictionary(step_dict)
    return combo_list

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
            #linalg_qpu = MPS(lnnize=True)
            linalg_qpu = MPS()
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
        #linalg_qpu = MPS(lnnize=True)
        linalg_qpu = MPS()
    else:
        raise ValueError(
            "Invalid value for qpu. Please select one of the three "+
            "following options: qlmass, python, c")
    #print("Following qpu will be used: {}".format(linalg_qpu))
    return linalg_qpu

def get_filelist(folder):
    """
    Look inside of a folder for files delete final part and get
    non repeated files
    """
    filelist = os.listdir(folder)
    filelist = set(map(lambda x: re.sub(r"_.*$", "", x), filelist))
    filelist = [folder + i for i in filelist]
    return filelist

def get_info_basefn(base_fn):
    depth = int(re.findall(r"depth_(.*)_qpu", base_fn)[0])
    nqubits = int(re.findall(r"nqubits_(.*)_depth_", base_fn)[0])
    ansatz = re.findall(r"ansatz_(.*)_nqubits", base_fn)[0]
    return depth, nqubits, ansatz

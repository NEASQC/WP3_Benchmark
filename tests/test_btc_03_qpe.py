import os
import shutil
import sys
import pandas as pd
import numpy as np

l_sys = sys.path
l_path = l_sys[['tests' in i for i in l_sys].index(True)]
l_path = l_path.replace("tests", '')
l_path = l_path + "/tnbs/"#BTC_03_QPE/"
sys.path.append(l_path)
sys.path.append(l_path+"BTC_03_QPE")
from BTC_03_QPE.my_benchmark_execution import KERNEL_BENCHMARK
from get_qpu import get_qpu


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

def test_qpe(): 

    kernel_configuration = {
        "angles" : ["random", 'exact'],
        "auxiliar_qbits_number" : [8],
        "qpu" : "c", #qlmass, python, c
        "fidelity_error" : None,
        "ks_error" : None,
        "time_error": None
    }
    kernel_configuration.update({"qpu": get_qpu(kernel_configuration['qpu'])})

    benchmark_arguments = {
        #Pre benchmark sttuff
        "pre_benchmark": True,
        "pre_samples": None,
        "pre_save": True,
        #Saving stuff
        "save_append" : True,
        "saving_folder": "./Results/",
        "benchmark_times": "kernel_times_benchmark.csv",
        "csv_results": "kernel_benchmark.csv",
        "summary_results": "kernel_SummaryResults.csv",
        #Computing Repetitions stuff
        "alpha": None,
        "min_meas": None,
        "max_meas": 10,
        #List number of qubits tested
        "list_of_qbits": [6],
    }

    benchmark_arguments.update({"kernel_configuration": kernel_configuration})
    folder = create_folder(benchmark_arguments["saving_folder"])
    kernel_bench = KERNEL_BENCHMARK(**benchmark_arguments)
    kernel_bench.exe()
    filename = "./Results/kernel_SummaryResults.csv"
    index_columns = [0, 1, 2, 3, 4, 5]
    a = pd.read_csv(filename, header=[0, 1], index_col=index_columns)
    a= a.reset_index()
    assert((a[a['angle_method']=="exact"]['fidelity']['mean'] > 0.999).all())
    assert((a[a['angle_method']=="random"]['KS']['mean'] < 0.05).all())
    shutil.rmtree(folder)

#test_qpe()

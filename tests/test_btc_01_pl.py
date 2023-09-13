import os
import shutil
import sys
import pandas as pd
import numpy as np

l_sys = sys.path
l_path = l_sys[['tests' in i for i in l_sys].index(True)]
l_path = l_path.replace("tests", '')
l_path = l_path + "tnbs/BTC_01_PL"
sys.path.append(l_path)
from my_benchmark_execution import KERNEL_BENCHMARK as PL_CLASS

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

def test_pl(): 
    kernel_configuration = {
        "load_method" : "multiplexor",
        "qpu" : "c", #python, qlmass, default
        "relative_error": None,
        "absolute_error": None
    }
    name = "PL_{}".format(kernel_configuration["load_method"])
    print(os.getcwdb())
                                                                  
    benchmark_arguments = {
        #Pre benchmark configuration
        "pre_benchmark": True,
        "pre_samples": None,
        "pre_save": False,
        #Saving configuration
        "save_append" : True,
        "saving_folder": "Results_PL/",
        "benchmark_times": "{}_times_benchmark.csv".format(name),
        "csv_results": "{}_benchmark.csv".format(name),
        "summary_results": "{}_SummaryResults.csv".format(name),
        #Computing Repetitions configuration
        "alpha": None,
        "min_meas": 10,
        "max_meas": 15,
        #List number of qubits tested
        "list_of_qbits": [4],
    }

    benchmark_arguments.update({"kernel_configuration": kernel_configuration})
    folder = create_folder(benchmark_arguments["saving_folder"])
    ae_bench = PL_CLASS(**benchmark_arguments)
    ae_bench.exe()
    filename = folder + benchmark_arguments["summary_results"]
    a = pd.read_csv(filename, header=[0, 1], index_col=[0, 1])
    print(100* list(a['KS']['mean'])[0])
    assert(100* list(a['KS']['mean'])[0] < 1.0)

    shutil.rmtree(folder)
#test_pl()

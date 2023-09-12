import os
import shutil
import sys
import pandas as pd
import numpy as np
import re

folder = os.getcwd()
folder = re.sub(r"WP3_Benchmark/.*", "WP3_Benchmark/", folder)
folder = folder + "tnbs/BTC_02_AE"
sys.path.append(folder)
from my_benchmark_execution import KERNEL_BENCHMARK 


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

def test_ae_iqae(): 

    kernel_configuration ={
        "ae_type": "IQAE",
        "epsilon": 0.001,
        "alpha": 0.05, # This the alpha for the AE algorithm
        "multiplexor":  True,
        "shots": 1000,
        "qpu": "c",
        "integrals": [0]
    }
    name = "AE_{}".format(kernel_configuration["ae_type"])

    benchmark_arguments = {
        #Pre benchmark configuration
        "pre_benchmark": True,
        "pre_samples": None,
        "pre_save": True,
        #Saving configuration
        "save_append" : True,
        "saving_folder": "./Results/",
        "benchmark_times": "{}_times_benchmark.csv".format(name),
        "csv_results": "{}_benchmark.csv".format(name),
        "summary_results": "{}_SummaryResults.csv".format(name),
        #Computing Repetitions configuration
        "alpha": None,
        "min_meas": None,
        "max_meas": 10,
        "relative_error": None,
        "absolute_error": None,
        #List number of qubits tested
        "list_of_qbits": [4],
    }

    benchmark_arguments.update({"kernel_configuration": kernel_configuration})
    folder = create_folder(benchmark_arguments["saving_folder"])
    ae_bench = KERNEL_BENCHMARK(**benchmark_arguments)
    ae_bench.exe()
    filename = folder + benchmark_arguments["summary_results"]
    a = pd.read_csv(filename, header=[0, 1], index_col=[0, 1])
    #print(a["absolute_error_sum"]["mean"])
    #print(100* list(a['KS']['mean'])[0])
    assert(100* list(a['absolute_error_sum']['mean'])[0] < 1.0)
    shutil.rmtree(folder)

def test_ae_mlae(): 

    kernel_configuration ={
        "ae_type": "MLAE",
        "schedule": [
                [1, 2, 4, 8, 12],
                [100, 100, 100, 100, 100]
         ],
        "delta": 1.0e-7,
        "ns": 10000,
        "mcz_qlm": False,
        "multiplexor":  True,
        "shots": 1000,
        "qpu": "c",
        "integrals": [0]
    }
    name = "AE_{}".format(kernel_configuration["ae_type"])

    benchmark_arguments = {
        #Pre benchmark configuration
        "pre_benchmark": True,
        "pre_samples": None,
        "pre_save": True,
        #Saving configuration
        "save_append" : True,
        "saving_folder": "./Results/",
        "benchmark_times": "{}_times_benchmark.csv".format(name),
        "csv_results": "{}_benchmark.csv".format(name),
        "summary_results": "{}_SummaryResults.csv".format(name),
        #Computing Repetitions configuration
        "alpha": None,
        "min_meas": None,
        "max_meas": 10,
        "relative_error": None,
        "absolute_error": None,
        #List number of qubits tested
        "list_of_qbits": [4],
    }

    benchmark_arguments.update({"kernel_configuration": kernel_configuration})
    folder = create_folder(benchmark_arguments["saving_folder"])
    ae_bench = KERNEL_BENCHMARK(**benchmark_arguments)
    ae_bench.exe()
    filename = folder + benchmark_arguments["summary_results"]
    a = pd.read_csv(filename, header=[0, 1], index_col=[0, 1])
    #print(a["absolute_error_sum"]["mean"])
    #print(100* list(a['KS']['mean'])[0])
    assert(100* list(a['absolute_error_sum']['mean'])[0] < 1.0)
    shutil.rmtree(folder)

def test_ae_rqae(): 

    kernel_configuration ={
        "ae_type": "RQAE",
        "epsilon": 0.01,
        "gamma": 0.05,
        "q": 1.2,

        "mcz_qlm": False,
        "multiplexor":  True,
        "shots": 1000,
        "qpu": "c",
        "integrals": [0, 1]
    }
    name = "AE_{}".format(kernel_configuration["ae_type"])

    benchmark_arguments = {
        #Pre benchmark configuration
        "pre_benchmark": True,
        "pre_samples": None,
        "pre_save": True,
        #Saving configuration
        "save_append" : True,
        "saving_folder": "./Results/",
        "benchmark_times": "{}_times_benchmark.csv".format(name),
        "csv_results": "{}_benchmark.csv".format(name),
        "summary_results": "{}_SummaryResults.csv".format(name),
        #Computing Repetitions configuration
        "alpha": None,
        "min_meas": None,
        "max_meas": 10,
        "relative_error": None,
        "absolute_error": None,
        #List number of qubits tested
        "list_of_qbits": [4],
    }

    benchmark_arguments.update({"kernel_configuration": kernel_configuration})
    folder = create_folder(benchmark_arguments["saving_folder"])
    ae_bench = KERNEL_BENCHMARK(**benchmark_arguments)
    ae_bench.exe()
    filename = folder + benchmark_arguments["summary_results"]
    a = pd.read_csv(filename, header=[0, 1], index_col=[0, 1])
    print(a["absolute_error_sum"]["mean"])
    assert((a["absolute_error_sum"]["mean"] < 0.01).all())
    #print(100* list(a['KS']['mean'])[0])
    #assert(100* list(a['absolute_error_sum']['mean'])[0] < 1.0)
    shutil.rmtree(folder)

#test_ae_iqae()
#test_ae_mlae()
#test_ae_rqae()

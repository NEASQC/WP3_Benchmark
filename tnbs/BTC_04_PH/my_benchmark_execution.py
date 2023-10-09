"""
This module execute a complete BTC of the PH kernel 
"""

import sys
import os
import ast
import logging
from datetime import datetime
from copy import deepcopy
import pandas as pd

l_sys = sys.path
l_path = l_sys[['BTC_04_PH' in i for i in l_sys].index(True)]
sys.path.append(l_path+'/PH')

from PH.ansatzes import ansatz_selector, angles_ansatz01
from PH.execution_ph import PH_EXE
from PH.utils_ph import get_qpu


logging.basicConfig(
    format='%(asctime)s-%(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO
    #level=logging.DEBUG
)
logger = logging.getLogger('__name__')

def build_iterator(**kwargs):
    """
    For building the iterator of the benchmark
    """
    import itertools as it

    list4it = [
        kwargs["list_of_qbits"],
        kwargs["kernel_configuration"]["depth"]
    ]

    iterator = it.product(*list4it)

    return list(iterator)

def run_code(iterator_step, repetitions, stage_bench, **kwargs):
    """
    For configuration and execution of the benchmark kernel.

    Parameters
    ----------

    iterator_step : tuple
        tuple with elements from iterator built from build_iterator.
    repetitions : list
        number of repetitions for each execution
    stage_bench : str
        benchmark stage. Only: benchmark, pre-benchamrk
    kwargs : keyword arguments
        for configuration of the benchmark kernel

    Returns
    _______

    metrics : pandas DataFrame
        DataFrame with the desired metrics obtained for the integral computation
    save_name : string
        Desired name for saving the results of the execution

    """

    if stage_bench not in ["benchmark", "pre-benchmark"]:
        raise ValueError(
            "Valid values for stage_bench: benchmark or pre-benchmark")
    if repetitions is None:
        raise ValueError("samples CAN NOT BE None")
    #Here the code for configuring and execute the benchmark kernel
    kernel_configuration = deepcopy(kwargs.get("kernel_configuration", None))
    del kernel_configuration["gse_error"]
    del kernel_configuration["time_error"]
    del kernel_configuration["depth"]
    if kernel_configuration is None:
        raise ValueError("kernel_configuration can not be None")
    # Configuring kernel

    nqubits = str(iterator_step[0]).zfill(2)
    depth = str(iterator_step[1])
    logger.info("Creating ansatz circuit")
    ansatz_conf = {
        "nqubits" :int(nqubits),
        "depth" : int(depth),
    }
    circuit = ansatz_selector("simple01", **ansatz_conf)
    # Formating Parameters
    conf_files_folder = kernel_configuration.get("conf_files_folder", None)
    if conf_files_folder is None:
        conf_files_folder = "configuration_files"
    base_fn = conf_files_folder + "/nqubits_{}_depth_{}".format(nqubits, depth)
    param_file = base_fn + "_parameters.csv"
    logger.info("Loading Parameters from: %s", param_file)

    parameters_pdf = pd.read_csv(param_file, sep=";", index_col=0)
    circuit, _ = angles_ansatz01(circuit, parameters_pdf)

    # Loading PH Pauli decomposition
    pauli_file = base_fn + "_pauli.csv"
    logger.info("Loading PH Pauli decomposition from: %s", pauli_file)
    # Loading Pauli
    pauli_pdf = pd.read_csv(pauli_file, sep=";", index_col=0)
    affected_qubits = [ast.literal_eval(i_) for i_ in list(pauli_pdf["Qbits"])]
    pauli_pdf["Qbits"] = affected_qubits

    # Executing VQE step
    logger.info("Executing VQE step")
    nb_shots = kernel_configuration.get("nb_shots", None)
    if nb_shots is None:
        nb_shots = 10000
    truncation = kernel_configuration.get("truncation", None)
    t_inv = kernel_configuration.get("t_inv", None)
    if t_inv is None:
        t_inv = True

    vqe_conf = {
        "qpu" : get_qpu(kernel_configuration["qpu_ph"]),
        "nb_shots": nb_shots,#kernel_configuration["nb_shots"],
        "truncation": truncation, #kernel_configuration["truncation"],
        "t_inv": t_inv,#kernel_configuration["t_inv"],
        "filename": None,
        "save": False,
    }
    list_of_metrics = []
    for i in range(repetitions):
        exe_ph = PH_EXE(circuit, pauli_pdf, int(nqubits), **vqe_conf)
        exe_ph.run()
        pdf_info = pd.DataFrame(
            [int(nqubits), int(depth)], index=["nqubits", "depth"]).T
        step = pd.DataFrame.from_dict(vqe_conf, orient="index").T
        list_ = [
            pdf_info,
            step,
            exe_ph.pdf_result
        ]
        pdf_info = pd.concat(list_, axis=1)
        list_of_metrics.append(pdf_info)
    metrics = pd.concat(list_of_metrics)
    metrics.reset_index(drop=True, inplace=True)
    metrics["elapsed_time"] = metrics["observable_time"] + \
        metrics["quantum_time"]
    if stage_bench == "pre-benchmark":
        # Name for storing Pre-Benchmark results
        save_name = "pre_benchmark_nq_{}_depth_{}.csv".format(
            iterator_step[0],
            iterator_step[1]
            )
    if stage_bench == "benchmark":
        # Name for storing Benchmark results
        save_name = kwargs.get("csv_results")
        #save_name = "pre_benchmark_step_{}.csv".format(n_qbits)
    return metrics, save_name

def compute_samples(**kwargs):
    """
    This function computes the number of executions of the benchmark
    for ensuring an error r with a confidence level of alpha

    Parameters
    ----------

    kwargs : keyword arguments
        For configuring the sampling computation

    Returns
    _______

    samples : pandas DataFrame
        DataFrame with the number of executions for each integration interval

    """

    #Configuration for sampling computations

    #Desired Confidence level
    alpha = kwargs.get("alpha", 0.05)
    if alpha is None:
        alpha = 0.05
    metrics = kwargs.get("pre_metrics")
    bench_conf = kwargs.get("kernel_configuration")

    #Code for computing the number of samples for getting the desired
    #statististical significance. Depends on benchmark kernel

    from scipy.stats import norm
    zalpha = norm.ppf(1-(alpha/2)) # 95% of confidence level

    # Error expected for the Groud State Energy
    error_gse = bench_conf.get("gse_error", 0.01)
    if error_gse is None:
        error_gse = 0.01
    std_ = metrics[["gse"]].std()
    samples_gse = (zalpha * std_ / error_gse) ** 2

    # Relative Error for elpased time
    time_error = bench_conf.get("time_error", 0.05)
    if time_error is None:
        time_error = 0.05
    mean_time = metrics[["elapsed_time"]].mean()
    std_time = metrics[["elapsed_time"]].std()
    samples_time = (zalpha * std_time / (time_error * mean_time)) ** 2

    #Maximum number of sampls will be used
    samples_ = pd.Series(pd.concat([samples_time, samples_gse]).max())

    #Apply lower and higher limits to samples
    #Minimum and Maximum number of samples
    min_meas = kwargs.get("min_meas", None)
    if min_meas is None:
        min_meas = 5
    max_meas = kwargs.get("max_meas", None)
    samples_.clip(upper=max_meas, lower=min_meas, inplace=True)
    samples_ = samples_[0].astype(int)
    return samples_

def summarize_results(**kwargs):
    """
    Create summary with statistics
    """

    folder = kwargs.get("saving_folder")
    csv_results = folder + kwargs.get("csv_results")

    #Code for summarize the benchamark results. Depending of the
    #kernel of the benchmark
    pdf = pd.read_csv(csv_results, index_col=0, sep=";")
    pdf["classic_time"] = pdf["elapsed_time"] - pdf["quantum_time"]
    pdf.fillna("None", inplace=True)
    group_columns = [
        "nqubits", "depth", "t_inv", "qpu",
        "nb_shots", "truncation"]
    metric_columns = ["gse", "elapsed_time", "quantum_time", "classic_time"]
    results = pdf.groupby(group_columns)[metric_columns].agg(
        ["mean", "std", "count"])
    results = results.replace('None', None)
    return results

class KERNEL_BENCHMARK:
    """
    Class for execute a Kernerl benchmark

    """


    def __init__(self, **kwargs):
        """

        Method for initializing the class

        """
        #Configurtion of benchmarked algorithm or routine
        self.kwargs = kwargs

        #Benchmark Configuration

        #Repetitions for pre benchmark step
        self.pre_samples = self.kwargs.get("pre_samples", None)
        if self.pre_samples is None:
            self.pre_samples = 10
        #Saving pre benchmark step results
        self.pre_save = self.kwargs.get("pre_save", True)
        #For executing or not the benchmark step
        self.pre_benchmark = self.kwargs.get("pre_benchmark", True)

        #Name for saving the pre benchmark step results
        self.save_name = self.kwargs.get("save_name", None)
        #Number of qbits
        self.list_of_qbits = self.kwargs.get("list_of_qbits", [4])

        save_type = self.kwargs.get("save_append", True)
        if save_type:
            self.save_type = "a"
        else:
            self.save_type = "w"

        #Create the iterator
        self.iterator = build_iterator(**self.kwargs)

        #Configure names for CSV files
        self.saving_folder = self.kwargs.get("saving_folder")
        self.benchmark_times = self.saving_folder + \
            self.kwargs.get("benchmark_times")
        self.csv_results = self.saving_folder + \
            self.kwargs.get("csv_results")
        self.summary_results = self.saving_folder + \
            self.kwargs.get("summary_results")

        #Attributes for metrics
        self.pre_metrics = None
        self.metrics = None


    def save(self, save, save_name, input_pdf, save_mode):
        """
        For saving panda DataFrames to csvs

        Parameters
        ----------

        save: bool
            For saving or not
        save_nam: str
            name for file
        input_pdf: pandas DataFrame
        save_mode: str
            saving mode: overwrite (w) or append (a)
        """
        if save:
            with open(save_name, save_mode) as f_pointer:
                input_pdf.to_csv(
                    f_pointer,
                    mode=save_mode,
                    header=f_pointer.tell() == 0,
                    sep=";"
                )

    def exe(self):
        """
        Execute complete Benchmark WorkFlow
        """
        start_time = datetime.now().astimezone().isoformat()
        for step_iterator in self.iterator:
            # print("n_qbits: {}".format(n_qbits))

            if self.pre_benchmark:
                print("\t Executing Pre-Benchmark")
                #Pre benchmark step
                pre_metrics, pre_save_name = run_code(
                    step_iterator, self.pre_samples, "pre-benchmark",
                    **self.kwargs
                )
                #For saving pre-benchmark step results
                pre_save_name = self.saving_folder + pre_save_name
                self.save(self.pre_save, pre_save_name, pre_metrics, "w")
                #Using pre benchmark results for computing the number of
                #repetitions
                self.kwargs.update({"pre_metrics": pre_metrics})
            #Compute needed samples for desired
            #statistical significance
            samples_ = compute_samples(**self.kwargs)
            print("\t Executing Benchmark Step")
            print("\t step samples: {}".format(samples_))
            metrics, save_name = run_code(
                step_iterator, samples_, "benchmark", **self.kwargs
            )
            save_name = self.saving_folder + save_name
            self.save(True, save_name, metrics, self.save_type)
        end_time = datetime.now().astimezone().isoformat()
        pdf_times = pd.DataFrame(
            [start_time, end_time],
            index=["StartTime", "EndTime"]
        ).T
        #Saving Time Info
        pdf_times.to_csv(self.benchmark_times)
        #Summarize Results
        results = summarize_results(**self.kwargs)
        results.to_csv(self.summary_results)


if __name__ == "__main__":


    #Anstaz
    depth = [1, 2, 3, 4]
    qpu_ph = "c"
    nb_shots = 0
    truncation = None

    kernel_configuration = {
        #Ansatz
        "conf_files_folder": None,
        "depth": depth,
        "t_inv": True,
        # Ground State Energy
        "qpu_ph" : qpu_ph,
        "nb_shots" : nb_shots,
        "truncation": truncation,
        # Saving
        "save": True,
        "folder": None,
        # Errors for confidence level
        "gse_error" : None,
        "time_error": None,
    }

    list_of_qbits = list(range(3, 9))
    benchmark_arguments = {
        #Pre benchmark sttuff
        "pre_benchmark": True,
        "pre_samples":  None,
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
        "max_meas": None,
        #List number of qubits tested
        "list_of_qbits": list_of_qbits,
    }

    #Configuration for the benchmark kernel
    benchmark_arguments.update({"kernel_configuration": kernel_configuration})
    kernel_bench = KERNEL_BENCHMARK(**benchmark_arguments)
    kernel_bench.exe()


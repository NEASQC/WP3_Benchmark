"""
Template for generic Benchmark Test Case Workflow
"""

import sys
import json
import copy
from datetime import datetime
from copy import deepcopy
import pandas as pd

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

    from PH.btc_ph import ph_btc, get_qpu, create_folder
    if stage_bench not in ["benchmark", "pre-benchmark"]:
        raise ValueError(
            "Valid values for stage_bench: benchmark or pre-benchmark")

    if repetitions is None:
        raise ValueError("samples CAN NOT BE None")
    #Here the code for configuring and execute the benchmark kernel
    kernel_configuration = deepcopy(kwargs.get("kernel_configuration", None))
    if kernel_configuration is None:
        raise ValueError("kernel_configuration can not be None")
    # Configuring kernel
    kernel_configuration.update({"nqubits": iterator_step[0]})
    kernel_configuration.update({"depth": iterator_step[1]})

    fold_name = "ansatz_{}_nqubits_{}_depth_{}_qpu_ansatz_{}_qpu_ph_{}".format(
        kernel_configuration["ansatz"],
        kernel_configuration["nqubits"],
        kernel_configuration["depth"],
        kernel_configuration["qpu_ansatz"],
        kernel_configuration["qpu_ph"])

    folder_name = kwargs["saving_folder"] + fold_name

    # Load QPU for solving ansatz
    kernel_configuration.update(
        {"qpu_ansatz": get_qpu(kernel_configuration["qpu_ansatz"])})
    # Load QPU for solving parent hamiltonian
    kernel_configuration.update(
        {"qpu_ph": get_qpu(kernel_configuration["qpu_ph"])})
    # Update and create the folder for intermediate storing
    kernel_configuration.update(
        {"folder": create_folder(folder_name)})


    # following keys are not mandatory for executing kernel
    del kernel_configuration["gse_error"]
    del kernel_configuration["time_error"]
    print(kernel_configuration)

    list_of_metrics = []
    #print(qpe_rz_dict)
    for i in range(repetitions):
        metric_step = ph_btc(**kernel_configuration)
        list_of_metrics.append(metric_step)

    metrics = pd.concat(list_of_metrics)
    metrics.reset_index(drop=True, inplace=True)

    if stage_bench == "pre-benchmark":
        # Name for storing Pre-Benchmark results
        save_name = "pre_benchmark_nq_{}_ansatz_{}_depth_{}.csv".format(
            iterator_step[0],
            kernel_configuration["ansatz"],
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
    samples_ = samples_.max().astype(int)
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
    pdf.drop(["save", "folder"], axis=1, inplace=True)
    # The angles are randomly selected. Not interesting for aggregation
    columns = pdf.columns
    columns = list(columns.drop(
        ["gse", "elapsed_time", "quantum_time", "classic_time"]))
    results = pdf.groupby(columns).agg(["mean", "std", "count"])
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
            self.save(self.save, save_name, metrics, self.save_type)
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

    import os
    import shutil

    ansatz = "simple02"
    qpu_ansatz = "mps"
    qpu_ph = "c"
    depth = [2, 3]

    kernel_configuration = {
        "ansatz": ansatz,
        "depth": depth,
        "qpu_ansatz" : qpu_ansatz,
        "qpu_ph" : qpu_ph,
        "nb_shots" : 0,
        "save": True,
        "folder": None,
        "gse_error" : None,
        "time_error": None
    }


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
        "max_meas": None,
        #List number of qubits tested
        "list_of_qbits": [4, 6],
    }

    #Configuration for the benchmark kernel
    benchmark_arguments.update({"kernel_configuration": kernel_configuration})
    kernel_bench = KERNEL_BENCHMARK(**benchmark_arguments)
    kernel_bench.exe()

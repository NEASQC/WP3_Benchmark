"""
Workflow configuration and execution for Benchmark Test Case of PL kernel
"""

import sys
import json
from datetime import datetime
import pandas as pd
from copy import deepcopy

def run_code(n_qbits, repetitions, **kwargs):
    """
    For configuration and execution of the benchmark kernel.

    Parameters
    ----------

    n_qbits : int
        number of qubits used for domain discretization
    repetitions : list
        number of repetitions for the integral
    kwargs : keyword arguments
        for configuration of the benchmark kernel

    Returns
    _______

    metrics : pandas DataFrame
        DataFrame with the desired metrics obtained for the integral computation

    """
    if n_qbits is None:
        raise ValueError("n_qbits CAN NOT BE None")
    if repetitions is None:
        raise ValueError("samples CAN NOT BE None")

    #Here the code for configuring and execute the benchmark kernel

    from load_probabilities import LoadProbabilityDensity
    kernel_configuration = deepcopy(kwargs.get("kernel_configuration", None))
    if kernel_configuration is None:
        raise ValueError("kernel_configuration can not be None")

    list_of_metrics = []
    kernel_configuration.update({"number_of_qbits": n_qbits})
    print(kernel_configuration)
    for i in range(repetitions[0]):
        prob_dens = LoadProbabilityDensity(**kernel_configuration)
        prob_dens.exe()
        list_of_metrics.append(prob_dens.pdf)
    metrics = pd.concat(list_of_metrics)
    metrics.reset_index(drop=True, inplace=True)

    return metrics

def compute_samples(**kwargs):
    """
    This functions computes the number of executions of the benchmark
    for assure an error r with a confidence of alpha

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

    #Desired Error in the benchmark metrics
    relative_error = kwargs.get("relative_error", None)
    if relative_error is None:
        relative_error = 0.1
    #Desired Confidence level
    alpha = kwargs.get("alpha", None)
    if alpha is None:
        alpha = 0.05
    #Minimum and Maximum number of samples
    min_meas = kwargs.get("min_meas", None)
    if min_meas is None:
        min_meas = 5
    max_meas = kwargs.get("max_meas", None)

    #Code for computing the number of samples for getting the desired
    #statististical significance. Depends on benchmark kernel

    from scipy.stats import norm
    #geting the metrics from pre-benchmark step
    metrics = kwargs.get("pre_metrics", None)

    #Compute mean and sd
    std_ = metrics.groupby("load_method").std()
    std_.reset_index(inplace=True)
    mean_ = metrics.groupby("load_method").mean()
    mean_.reset_index(inplace=True)
    #Metrics
    zalpha = norm.ppf(1-(alpha/2)) # 95% of confidence level
    #columns = ["KS", "KL", "elapsed_time"]
    columns = ["elapsed_time"]

    samples_ = (zalpha * std_[columns] / (relative_error * mean_[columns]))**2
    samples_ = samples_.max(axis=1).astype(int)
    samples_.name = "samples"

    #If user wants limit the number of samples
    samples_.clip(upper=max_meas, lower=min_meas, inplace=True)
    return list(samples_)

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
    pdf = pdf[
        ["n_qbits", "load_method", "KS", "KL", "chi2",
        "p_value", "elapsed_time", "quantum_time", "classic_time"]
    ]
    results = pdf.groupby(["load_method", "n_qbits"]).agg(
        ["mean", "std", "count"])

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
        self.pre_samples = self.kwargs.get("pre_samples", 10)
        #Saving pre benchmark step results
        self.pre_save = self.kwargs.get("pre_save", True)
        #For executing or not the benchmark step
        self.pre_benchmark = self.kwargs.get("pre_benchmark", True)

        #Name for saving the pre benchmark step results
        self.save_name = self.kwargs.get("save_name", None)
        #NNumber of qbits
        self.list_of_qbits = self.kwargs.get("list_of_qbits", [4])

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
                    sep=';'
                )

    def exe(self):
        """
        Execute complete Benchmark WorkFlow
        """
        start_time = datetime.now().astimezone().isoformat()
        for n_qbits in self.list_of_qbits:
            print("n_qbits: {}".format(n_qbits))

            if self.pre_benchmark:
                print("\t Executing Pre-Benchmark")
                #Pre benchmark step
                pre_metrics = run_code(
                    n_qbits, self.pre_samples, **self.kwargs
                )
                #For saving pre-benchmark step results
                pre_save_name = self.saving_folder + \
                    "pre_benchmark_step_{}.csv".format(n_qbits)
                self.save(self.pre_save, pre_save_name, pre_metrics, "w")
                #Using pre benchmark results for computing the number of
                #repetitions
                self.kwargs.update({"pre_metrics": pre_metrics})

            #Compute needed samples for desired
            #statistical significance
            samples_ = compute_samples(**self.kwargs)
            print("\t step samples: {}".format(samples_))
            metrics = run_code(
                n_qbits, samples_, **self.kwargs
            )
            self.save(self.save, self.csv_results, metrics, "a")
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

    kernel_configuration = {
        "load_method" : "multiplexor",
        "qpu" : "c", #python, qlmass, default
    }
    name = "PL_{}".format(kernel_configuration["load_method"])

    benchmark_arguments = {
        #Pre benchmark configuration
        "pre_benchmark": True,
        "pre_samples": [10],
        "pre_save": True,
        #Saving configuration
        "saving_folder": "./Results/",
        "benchmark_times": "{}_times_benchmark.csv".format(name),
        "csv_results": "{}_benchmark.csv".format(name),
        "summary_results": "{}_SummaryResults.csv".format(name),
        #Computing Repetitions configuration
        "relative_error": None,
        "alpha": None,
        "min_meas": None,
        "max_meas": None,
        #List number of qubits tested
        "list_of_qbits": [4, 6, 8],
    }

    #Configuration for the benchmark kernel
    benchmark_arguments.update({"kernel_configuration": kernel_configuration})
    ae_bench = KERNEL_BENCHMARK(**benchmark_arguments)
    ae_bench.exe()


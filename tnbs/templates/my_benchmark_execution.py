"""
Template for generic Benchmark Test Case Workflow
"""

import sys
import json
import copy
from datetime import datetime
import pandas as pd


def build_iterator(**kwargs):
    """
    For building the iterator of the benchmark
    """

    iterator = [(2,'dummy'), (2, 'dummy2'), (4, 'dummy'), (2, 'dummy4')]
    return iterator


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
    # if n_qbits is None:
    #     raise ValueError("n_qbits CAN NOT BE None")

    if stage_bench not in ['benchmark', 'pre-benchmark']:
        raise ValueError(
            "Valid values for stage_bench: benchmark or pre-benchmark'")

    if repetitions is None:
        raise ValueError("samples CAN NOT BE None")

    #Here the code for configuring and execute the benchmark kernel
    variable_0 = iterator_step[0]
    variable_1 = iterator_step[1]

    # Loop over the BTC step repetitions
    for i in range(repetitions):
        continue

    metrics = pd.DataFrame()

    # Deals with the file names for storing metrics
    if stage_bench == 'pre-benchmark':
        # Name for storing Pre-Benchmark results
        save_name = "pre_benchmark.csv"
    if stage_bench == 'benchmark':
        # Name for storing Benchmark results
        save_name = kwargs.get('csv_results')
        #save_name = "pre_benchmark_step_{}.csv".format(n_qbits)
    return metrics, save_name

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

    #Desired Confidence level
    alpha = kwargs.get("alpha", 0.05)


    #Code for computing the number of samples for getting the desired
    #statististical significance. Depends on benchmark kernel
    samples_ = pd.Series([100, 100])
    samples_.name = "samples"

    #If user wants limit the number of samples

    #Minimum and Maximum number of samples
    min_meas = kwargs.get("min_meas", 5)
    max_meas = kwargs.get("max_meas", None)
    samples_.clip(upper=max_meas, lower=min_meas, inplace=True)
    return list(samples_)

def summarize_results(**kwargs):
    """
    Create summary with statistics
    """

    folder = kwargs.get("saving_folder")
    csv_results = kwargs.get("csv_results")

    #Code for summarize the benchamark results. Depending of the
    #kernel of the benchmark

    results = pd.DataFrame()

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

        save_type = self.kwargs.get("save_append", True)
        if save_type:
            self.save_type = 'a'
        else:
            self.save_type = 'w'

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
                    sep=';'
                )

    def exe(self):
        """
        Execute complete Benchmark WorkFlow
        """
        start_time = datetime.now().astimezone().isoformat()
        for step_iterator in self.iterator:
            #print("n_qbits: {}".format(n_qbits))

            if self.pre_benchmark:
                print("\t Executing Pre-Benchmark")
                #Pre benchmark step
                pre_metrics, pre_save_name = run_code(
                    step_iterator, self.pre_samples, 'pre-benchmark',
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
                step_iterator, samples_, 'benchmark', **self.kwargs
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

    benchmark_arguments = {
        #Pre benchmark sttuff
        "pre_benchmark": False,
        "pre_samples": 10, 
        "pre_save": True,
        #Saving stuff
        "saving_folder": "./Results/",
        "benchmark_times": "kernel_times_benchmark.csv",
        "csv_results": "kernel_benchmark.csv",
        "summary_results": "kernel_SummaryResults.csv",
        #Computing Repetitions stuff
        "alpha": 0.05,
        "min_meas": 5,
        "max_meas": 10,
        #List number of qubits tested
        "list_of_qbits": [4],#, 6, 8],
    }

    kernel_bench = KERNEL_BENCHMARK(**benchmark_arguments)
    kernel_bench.exe()

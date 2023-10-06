"""
This module execute a complete BTC of the PL kernel 
"""

import sys
import json
from datetime import datetime
from copy import deepcopy
import pandas as pd

l_sys = sys.path
l_path = l_sys[['BTC_01' in i for i in l_sys].index(True)]
sys.path.append(l_path+'/PL')
from PL.load_probabilities import LoadProbabilityDensity, get_qpu

def build_iterator(**kwargs):
    """
    For building the iterator of the benchmark
    """

    iterator = [tuple([i]) for i in kwargs['list_of_qbits']]
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
        DataFrame with the desired metrics obtained for the
        integral computation
    save_name : string
        Desired name for saving the results of the execution

    """
    #if n_qbits is None:
    #    raise ValueError("n_qbits CAN NOT BE None")

    if stage_bench not in ['benchmark', 'pre-benchmark']:
        raise ValueError(
            "Valid values for stage_bench: benchmark or pre-benchmark'")

    if repetitions is None:
        raise ValueError("repetitions CAN NOT BE None")

    #Here the code for configuring and execute the benchmark kernel
    kernel_configuration = deepcopy(kwargs.get("kernel_configuration", None))
    if kernel_configuration is None:
        raise ValueError("kernel_configuration can not be None")

    # Here we built the dictionary for the LoadProbabilityDensity class
    n_qbits = iterator_step[0]
    list_of_metrics = []
    kernel_configuration.update({"number_of_qbits": n_qbits})
    kernel_configuration.update({"qpu": get_qpu(kernel_configuration['qpu'])})
    print(kernel_configuration)
    for i in range(repetitions):
        prob_dens = LoadProbabilityDensity(**kernel_configuration)
        prob_dens.exe()
        list_of_metrics.append(prob_dens.pdf)
    metrics = pd.concat(list_of_metrics)
    metrics.reset_index(drop=True, inplace=True)

    if stage_bench == 'pre-benchmark':
        # Name for storing Pre-Benchmark results
        save_name = "pre_benchmark_step_{}.csv".format(n_qbits)
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
    from scipy.stats import norm

    #Configuration for sampling computations

    #Desired Confidence level
    alpha = kwargs.get("alpha", None)
    if alpha is None:
        alpha = 0.05
    zalpha = norm.ppf(1-(alpha/2)) # 95% of confidence level

    #geting the metrics from pre-benchmark step
    metrics = kwargs.get("pre_metrics", None)
    bench_conf = kwargs.get('kernel_configuration')

    #Code for computing the number of samples for getting the desired
    #statististical significance. Depends on benchmark kernel

    #Desired Relative Error for the elapsed Time
    relative_error = bench_conf.get("relative_error", None)
    if relative_error is None:
        relative_error = 0.05
    # Compute samples for Elapsed Time
    samples_t = (zalpha * metrics[['elapsed_time']].std() / \
        (relative_error * metrics[['elapsed_time']].mean()))**2

    #Desired Absolute Error for KS and KL metrics
    absolute_error = bench_conf.get("absolute_error", None)
    if absolute_error is None:
        absolute_error = 1e-4
    std_metrics = metrics[['KS', 'KL']].std()
    samples_m = (zalpha * std_metrics / absolute_error) ** 2

    #Maximum number of sampls will be used
    samples_ = pd.Series(pd.concat([samples_t, samples_m]).max())

    #Apply lower and higher limits to samples
    #Minimum and Maximum number of samples
    min_meas = kwargs.get("min_meas", None)
    if min_meas is None:
        min_meas = 5
    max_meas = kwargs.get("max_meas", None)
    samples_.clip(upper=max_meas, lower=min_meas, inplace=True)
    print(samples_)
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
    columns = ["n_qbits", "load_method", "KS", "KL", "chi2", "p_value", \
        "elapsed_time", "quantum_time", "classic_time"]
    pdf = pdf[columns]
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

    kernel_configuration = {
        "load_method" : "multiplexor",
        "qpu" : "c", #python, qlmass, default
        "relative_error": None,
        "absolute_error": None
    }
    name = "PL_{}".format(kernel_configuration["load_method"])

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
        "max_meas": None,
        #List number of qubits tested
        "list_of_qbits": [4, 6, 8, 10],
    }

    #Configuration for the benchmark kernel
    benchmark_arguments.update({"kernel_configuration": kernel_configuration})
    ae_bench = KERNEL_BENCHMARK(**benchmark_arguments)
    ae_bench.exe()


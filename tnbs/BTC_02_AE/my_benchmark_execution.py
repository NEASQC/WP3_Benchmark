"""
This module execute a complete BTC of the AE kernel
"""

import json
from datetime import datetime
import pandas as pd
from QQuantLib.ae_sine_integral import sine_integral

def build_iterator(**kwargs):
    """
    For building the iterator of the benchmark
    """
    import itertools as it

    list4int = [
        kwargs['list_of_qbits'],
        kwargs['kernel_configuration']['integrals'],
    ]

    iterator = it.product(*list4int)
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
    #if n_qbits is None:
    #    raise ValueError("n_qbits CAN NOT BE None")
    if stage_bench not in ['benchmark', 'pre-benchmark']:
        raise ValueError(
            "Valid values for stage_bench: benchmark or pre-benchmark'")
    if repetitions is None:
        raise ValueError("samples CAN NOT BE None")

    n_qbits = iterator_step[0]
    interval = iterator_step[1]

    #Here the code for configuring and execute the benchmark kernel
    ae_configuration = kwargs.get("kernel_configuration")

    columns = [
        "interval", "n_qbits", "IntegralAbsoluteError", "oracle_calls",
        "elapsed_time", "run_time", "quantum_time"
    ]

    list_of_metrics = []
    for i in range(repetitions):
        metrics = sine_integral(n_qbits, interval, ae_configuration)
        list_of_metrics.append(metrics)
    metrics = pd.concat(list_of_metrics)
    metrics.reset_index(drop=True, inplace=True)

    if stage_bench == 'pre-benchmark':
        # Name for storing Pre-Benchmark results
        save_name = "pre_benchmark_nqubits_{}_integral_{}.csv".format(
            n_qbits, interval)
    if stage_bench == 'benchmark':
        # Name for storing Benchmark results
        save_name = kwargs.get('csv_results')
        #save_name = "pre_benchmark_step_{}.csv".format(n_qbits)
    return metrics[columns], save_name

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

    samples : int
        Computed number of executions for desired significance

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
    #getting the configuration of the algorithm and kernel
    bench_conf = kwargs.get('kernel_configuration')

    #Code for computing the number of samples for getting the desired
    #statististical significance. Depends on benchmark kernel

    #Desired Relative Error for the elapsed Time
    relative_error = kwargs.get("relative_error", None)
    if relative_error is None:
        relative_error = 0.05
    # Compute samples for realtive error metrics:
    # Elapsed Time and Oracle Calls
    samples_re = (zalpha * metrics[['elapsed_time', 'oracle_calls']].std() / \
        (relative_error * metrics[['elapsed_time', 'oracle_calls']].mean()))**2

    #Desired Absolute Error.
    absolute_error = kwargs.get("absolute_error", None)
    if absolute_error is None:
        absolute_error = 1e-4
    samples_ae = (zalpha * metrics[['IntegralAbsoluteError']].std() \
        / absolute_error) ** 2

    #Maximum number of sampls will be used
    samples_ = pd.Series(pd.concat([samples_re, samples_ae]).max())

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

    #Code for summarize the benchamark results. Depending of the
    #kernel of the benchmark
    folder = kwargs.get("saving_folder")
    csv_results = folder + kwargs.get("csv_results")

    #results = pd.DataFrame()
    pdf = pd.read_csv(csv_results, index_col=0, sep=";")
    pdf["classic_time"] = pdf["elapsed_time"] - pdf["quantum_time"]
    results = pdf.groupby(["interval", "n_qbits"]).describe()

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
    import os
    import sys
    sys.path.append("../")
    from qpu.select_qpu import select_qpu
    from qpu.benchmark_utils import combination_for_list
    from qpu.benchmark_utils import create_ae_pe_solution

    ############## CONFIGURE THE BTC  ###################

    # Path for AE JSON file configuration
    #ae_json_file = "jsons/integral_mlae_configuration.json"
    ae_json_file = "jsons/integral_iqae_configuration.json"
    # For setting the AE configuration
    id_ae = 0

    # Path for QPU JSON file configuration
    qpu_json_file = "../qpu/qpu_ideal.json"
    # qpu_json_file = "qpu_noisy.json"
    # For setting the qpu configuration
    id_qpu = 0

    # Setting the integral interval
    #list_of_integral_intervals = [0, 1]
    list_of_integral_intervals = [0]
    # Setting the list of qubits for domain discretization
    list_of_qbits = [6]

    ############## CONFIGURE THE BTC  ###################

    ############# LOAD the JSON FILES ###################
    # Setting the AE algorithm configuration
    with open(ae_json_file) as json_file:
        ae_cfg = json.load(json_file)
    # BE AWARE only one AE configuration MUST BE provided
    ae_problem = combination_for_list(ae_cfg)[id_ae]

    # Setting the QPU configuration
    with open(qpu_json_file) as json_file:
        noisy_cfg = json.load(json_file)
    # BE AWARE only one QPU configuration MUST BE provided
    qpu_conf = combination_for_list(noisy_cfg)[id_qpu]
    ae_problem.update(qpu_conf)


    ############## CONFIGURE THE BENCHMARK EXECUTION  #################
    ae_problem.update({
        "integrals": list_of_integral_intervals
    })

    AE = ae_problem["ae_type"]
    # Configure the save Folder and the name of the files
    saving_folder = "./{}_Results/".format(AE)
    benchmark_times = "{}_times_benchmark.csv".format(AE)
    csv_results = "{}_benchmark.csv".format(AE)
    summary_results = "{}_SummaryResults.csv".format(AE)

    benchmark_arguments = {
        #Pre benchmark sttuff
        "pre_benchmark": True,
        "pre_samples": None,
        "pre_save": True,
        #Saving stuff
        "save_append" : True,
        "saving_folder": saving_folder,
        "benchmark_times": benchmark_times,
        "csv_results": csv_results,
        "summary_results": summary_results,
        #Computing Repetitions stuff
        "alpha": None,
        "relative_error": None,
        "absolute_error": None,
        "min_meas": None,
        "max_meas": None,
        #List number of qubits tested
        "list_of_qbits": list_of_qbits,
    }
    ############## CONFIGURE THE BENCHMARK EXECUTION  #################


    json_object = json.dumps(ae_problem)
    if not os.path.exists(benchmark_arguments["saving_folder"]):
        os.mkdir(benchmark_arguments["saving_folder"])
    #Writing the AE algorithm configuration
    conf_file = benchmark_arguments["saving_folder"] + \
        "benchmark_ae_conf.json"
    with open(conf_file, "w") as outfile:
        outfile.write(json_object)
    # Store the QPU configuration
    qpu_file = benchmark_arguments["saving_folder"] + \
        "qpu_configuration.json"
    with open(qpu_file, "w") as outfile:
        json.dump(qpu_conf, outfile)
    #Added ae configuration
    ae_problem.update({"qpu":select_qpu(ae_problem)})
    benchmark_arguments.update({
        "kernel_configuration": ae_problem,
    })
    ae_bench = KERNEL_BENCHMARK(**benchmark_arguments)
    ae_bench.exe()

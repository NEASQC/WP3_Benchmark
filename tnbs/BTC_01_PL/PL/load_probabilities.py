"""
Mandatory code for softaware implemetation of the Benchmark Test Case
of PL kernel
"""

import time
import itertools
import numpy as np
import pandas as pd
from scipy.stats import entropy, kstest
from data_loading import get_theoric_probability, get_qlm_probability

class LoadProbabilityDensity:
    """
    Probability Loading
    """


    def __init__(self, **kwargs):
        """

        Method for initializing the class

        """

        self.n_qbits = kwargs.get("number_of_qbits", None)
        if self.n_qbits is None:
            error_text = "The number_of_qbits argument CAN NOT BE NONE."
            raise ValueError(error_text)
        self.load_method = kwargs.get("load_method", None)
        if self.load_method is None:
            error_text = "The load_method argument CAN NOT BE NONE."\
                "Select between: multiplexor, brute_force or KPTree"
            raise ValueError(error_text)
        # Set the QPU to use
        self.qpu = kwargs.get("qpu", None)
        if self.qpu is None:
            error_text = "Please provide a QPU."
            raise ValueError(error_text)

        self.data = None
        self.p_gate = None
        self.result = None
        self.circuit = None
        self.quantum_time = None
        self.elapsed_time = None
        #Distribution related attributes
        self.x_ = None
        self.data = None
        self.mean = None
        self.sigma = None
        self.step = None
        self.shots = None
        self.dist = None
        #Metric stuff
        self.ks = None
        self.kl = None
        self.fidelity = None
        self.pdf = None
        self.kl_pdf = None
        self.samples = None
        self.ks_scipy = None
        self.ks_pvalue = None

    def get_quantum_pdf(self):
        """
        Computing quantum probability density function
        """
        self.result, self.circuit, self.quantum_time = get_qlm_probability(
            self.data, self.load_method, self.shots, self.qpu)
        # For ordering the index using the Int_lsb
        self.result.reset_index(inplace=True)

    def get_theoric_pdf(self):
        """
        Computing theoretical probability densitiy function
        """
        self.x_, self.data, self.mean, self.sigma, \
            self.step, self.shots, self.dist = get_theoric_probability(self.n_qbits)

    def get_metrics(self):
        """
        Computing Metrics
        """
        #Kolmogorov-Smirnov
        # Transform from state to x value
        self.result["x"] = self.x_[self.result["Int_lsb"]]
        # Compute cumulative distribution function of the quantum data
        self.result["CDF_quantum"] = self.result["Probability"].cumsum()
        # Obtain the cumulative distribution function of the
        # theoretical Gaussian distribution
        self.result["CDF_theorical"] = self.dist.cdf(self.result["x"])
        self.ks = np.abs(
            self.result["CDF_quantum"] - self.result["CDF_theorical"]).max()

        #Kullback-Leibler divergence
        epsilon = self.data.min() * 1.0e-5
        # Create pandas DF for KL computations
        self.kl_pdf = pd.merge(
            pd.DataFrame(
                [self.x_, self.data], index=["x", "p_th"]
            ).T,
            self.result[["x", "Probability"]],
            on=["x"], how="outer"
        ).fillna(epsilon)
        self.kl = entropy(
            self.kl_pdf["p_th"], self.kl_pdf["Probability"]
        )

        # For testing purpouses
        self.samples = list(itertools.chain(
            *self.result.apply(
                lambda x: [x["x"]] * int(round(
                    x["Probability"] * self.shots
                )),
                axis=1
            )
        ))
        ks_scipy = kstest(self.samples, self.dist.cdf)
        self.ks_scipy = ks_scipy.statistic
        self.ks_pvalue = ks_scipy.pvalue

    def exe(self):
        """
        Execution of workflow
        """
        #Create the distribution for loading
        tick = time.time()
        self.get_theoric_pdf()
        #Execute the quantum program
        self.get_quantum_pdf()
        self.get_metrics()
        tack = time.time()
        self.elapsed_time = tack - tick
        self.summary()

    def summary(self):
        """
        Pandas summary
        """
        self.pdf = pd.DataFrame()
        self.pdf["n_qbits"] = [self.n_qbits]
        self.pdf["load_method"] = [self.load_method]
        self.pdf["qpu"] = [self.qpu]
        self.pdf["mean"] = [self.mean]
        self.pdf["sigma"] = [self.sigma]
        self.pdf["step"] = [self.step]
        self.pdf["shots"] = [self.shots]
        self.pdf["KS"] = [self.ks]
        self.pdf["KL"] = [self.kl]
        self.pdf["elapsed_time"] = [self.elapsed_time]
        self.pdf["quantum_time"] = [self.quantum_time]


if __name__ == "__main__":
    import argparse
    import json
    from qpu.benchmark_utils import combination_for_list
    from qpu.select_qpu import select_qpu


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n_qbits",
        dest="n_qbits",
        type=int,
        help="Number of qbits for interval discretization.",
        default=None,
    )
    parser.add_argument(
        "-method",
        dest="method",
        type=str,
        help="For selecting the load method: multiplexor, brute_force, KPTree",
        default=None,
    )
    #QPU argument
    parser.add_argument(
        "-qpu",
        dest="qpu",
        type=str,
        default="python",
        help="QPU for simulation: See function get_qpu in get_qpu module",
    )
    parser.add_argument(
        "--count",
        dest="count",
        default=False,
        action="store_true",
        help="For counting elements on the list",
    )
    parser.add_argument(
        "--print",
        dest="print",
        default=False,
        action="store_true",
        help="For printing "
    )
    parser.add_argument(
        "-id",
        dest="id",
        type=int,
        help="For executing only one element of the list",
        default=None,
    )
    parser.add_argument(
        "-json_qpu",
        dest="json_qpu",
        type=str,
        default="qpu/qpu.json",
        help="JSON with the qpu configuration",
    )
    parser.add_argument(
        "--exe",
        dest="execution",
        default=False,
        action="store_true",
        help="For executing program",
    )
    args = parser.parse_args()
    with open(args.json_qpu) as json_file:
        noisy_cfg = json.load(json_file)
    final_list = combination_for_list(noisy_cfg)


    if args.count:
        print(len(final_list))
    if args.print:
        if args.id is not None:
            configuration = {
                "load_method" : args.method,
                "number_of_qbits": args.n_qbits,
                "qpu": final_list[args.id]
            }
            print(configuration)
        else:
            print(final_list)
    if args.execution:
        if args.id is not None:
            configuration = {
                "load_method" : args.method,
                "number_of_qbits": args.n_qbits,
                "qpu": select_qpu(final_list[args.id])
            }
            prob_dens = LoadProbabilityDensity(**configuration)
            prob_dens.exe()
            print(prob_dens.pdf)

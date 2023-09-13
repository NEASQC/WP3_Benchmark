"""
Mandatory code for softaware implemetation of the Benchmark Test Case
of PL kernel
"""

import sys
import os
import time
import re
import numpy as np
import pandas as pd
from scipy.stats import entropy, chisquare, chi2

folder = os.getcwd()
folder = re.sub(
    r"WP3_Benchmark/(?=WP3_Benchmark/)*.*","WP3_Benchmark/", folder)
sys.path.append(folder)
from tnbs.btc_01_pl.PL.data_loading import get_theoric_probability, get_qlm_probability, \
    get_qpu



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
        self.chi2 = None
        self.fidelity = None
        self.pvalue = None
        self.pdf = None
        self.observed_frecuency = None
        self.expected_frecuency = None

    def get_quantum_pdf(self):
        """
        Computing quantum probability density function
        """
        self.result, self.circuit, self.quantum_time = get_qlm_probability(
            self.data, self.load_method, self.shots, self.qpu)

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
        self.ks = np.abs(
            self.result["Probability"].cumsum() - self.data.cumsum()
        ).max()
        #Kullback-Leibler divergence
        epsilon = self.data.min() * 1.0e-5
        self.kl = entropy(
            self.data,
            np.maximum(epsilon, self.result["Probability"])
        )
        #Fidelity
        self.fidelity = self.result["Probability"] @ self.data / \
            (np.linalg.norm(self.result["Probability"]) * \
            np.linalg.norm(self.data))

        #Chi square
        self.observed_frecuency = np.round(
            self.result["Probability"] * self.shots, decimals=0)
        self.expected_frecuency = np.round(
            self.data * self.shots, decimals=0)
        try:
            self.chi2, self.pvalue = chisquare(
                f_obs=self.observed_frecuency,
                f_exp=self.expected_frecuency
            )
        except ValueError:
            self.chi2 = np.sum(
                (self.observed_frecuency - self.expected_frecuency) **2 / \
                    self.expected_frecuency
            )
            count = len(self.observed_frecuency)
            self.pvalue = chi2.sf(self.chi2, count -1)

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
        self.pdf["fidelity"] = [self.fidelity]
        self.pdf["chi2"] = [self.chi2]
        self.pdf["p_value"] = [self.pvalue]
        self.pdf["elapsed_time"] = [self.elapsed_time]
        self.pdf["quantum_time"] = [self.quantum_time]


if __name__ == "__main__":
    import argparse

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
        help="QPU for simulation: [qlmass, python, c]",
    )
    args = parser.parse_args()
    print(args)


    configuration = {
        "load_method" : args.method,
        "number_of_qbits": args.n_qbits,
        "qpu": get_qpu(args.qpu)
    }
    prob_dens = LoadProbabilityDensity(**configuration)
    prob_dens.exe()
    print(prob_dens.pdf)

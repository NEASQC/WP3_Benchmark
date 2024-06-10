"""
Class for executing QPE on a R_z^n operator


Author: Gonzalo Ferro
"""

import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import norm, entropy, chisquare, chi2
sys.path.append("../")

from QPE import rz_lib


def save(save, save_name, input_pdf, save_mode):
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

    

class QPE_RZ:
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
        self.auxiliar_qbits_number = kwargs.get("auxiliar_qbits_number", None)
        if self.auxiliar_qbits_number is None:
            error_text = "Provide the number of auxiliary qubits for QPE"
            raise ValueError(error_text)

        # Minimum angle measured by the QPE
        self.delta_theta = 4 * np.pi / 2 ** self.auxiliar_qbits_number

        angles = kwargs.get("angles", None)
        if type(angles) not in [str, float, list]:
            error_text = "Be aware! angles keyword can only be str" + \
                ", float or list "
            raise ValueError(error_text)

        self.angle_name = ''

        if isinstance(angles, str):
            if angles == 'random':
                self.angle_name = 'random'
                self.angles = [np.pi * np.random.random() \
                    for i in range(self.n_qbits)]
            elif angles == 'exact':
                # Here we compute the angles of the R_z^n operator for
                # obtaining exact eigenvalues in QPE. We begin in 0.5pi
                # and sum or rest randomly the minimum QPE measured
                # angle
                self.angle_name = 'exact'
                self.angles = []
                angle_0 = np.pi / 2.0
                for i_ in range(self.n_qbits):
                    angle_0 = angle_0 + (-1) ** np.random.randint(2) *\
                        self.delta_theta
                    self.angles.append(angle_0)
            else:
                error_text = "Be aware! If angles is str then only" + \
                    "can be random"
                raise ValueError(error_text)

        if isinstance(angles, float):
            self.angles = [angles for i in range(self.n_qbits)]

        if isinstance(angles, list):
            self.angles = angles
            if len(self.angles) != self.n_qbits:
                error_text = "Be aware! The number of elements in angles" + \
                    "MUST BE equal to the n_qbits"
                raise ValueError(error_text)

        # Set the QPU to use
        self.qpu = kwargs.get("qpu", None)
        if self.qpu is None:
            error_text = "Please provide a QPU."
            raise ValueError(error_text)

        # Shots for measuring the QPE circuit
        self.shots = kwargs.get("shots", None)
        if self.shots is not None:
            text = "BE AWARE! The keyword shots should be None because" +\
                "shots should be computed in function of the theoretical" +\
                "eigenvalues. You can only provide 0 for doing some testing" +\
                "in the class. 0 will imply complete simulation of QPE circuit"
            print(text)
            if self.shots != 0:
                error_text = "BE AWARE! The keyword shots must be None or 0"
                raise ValueError(error_text)


        # For storing classical eigenvalue distribution
        self.theorical_eigv = None
        self.theorical_eigv_dist = None
        # For storing quantum eigenvalue distribution
        self.quantum_eigv_dist = None
        # For storing attributes from CQPE class
        self.circuit = None
        self.quantum_time = None

        # Computing complete time of the process
        self.elapsed_time = None

        # Metric attributes
        self.ks = None
        self.fidelity = None

        # Pandas DataFrame for summary
        self.pdf = None

    def theoretical_distribution(self):
        """
        Computes the theoretical distribution of Rz eigenvalues
        """
        # Compute the complete eigenvalues
        self.theorical_eigv = rz_lib.rz_eigv(self.angles)
        # Compute the eigenvalue distribution using auxiliar_qbits_number
        self.theorical_eigv_dist = rz_lib.make_histogram(
            self.theorical_eigv['Eigenvalues'], self.auxiliar_qbits_number)
        if self.shots is None:
            # Compute the number of shots for QPE circuit
            self.shots = rz_lib.computing_shots(self.theorical_eigv)
        else:
            if self.shots != 0:
                self.shots = rz_lib.computing_shots(self.theorical_eigv)
            else:
                pass

    def quantum_distribution(self):
        """
        Computes the quantum distribution of Rz eigenvalues
        """
        self.quantum_eigv_dist, qpe_object = rz_lib.qpe_rz_qlm(
            self.angles,
            auxiliar_qbits_number=self.auxiliar_qbits_number,
            shots=self.shots,
            qpu=self.qpu

        )
        self.circuit = qpe_object.circuit
        self.quantum_time = qpe_object.quantum_times

    def get_metrics(self):
        """
        Computing Metrics
        """
        # Kolmogorov-Smirnov
        self.ks = np.abs(
            self.theorical_eigv_dist['Probability'].cumsum() \
                - self.quantum_eigv_dist['Probability'].cumsum()
        ).max()
        # Fidelity
        qv = self.quantum_eigv_dist['Probability']
        tv = self.theorical_eigv_dist['Probability']
        self.fidelity = qv @ tv / (np.linalg.norm(qv) * np.linalg.norm(tv))


    def exe(self):
        """
        Execution of workflow
        """
        tick = time.time()
        # Compute theoretical eigenvalues
        self.theoretical_distribution()
        # Computing eigenvalues using QPE
        self.quantum_distribution()
        # Compute the metrics
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
        self.pdf["aux_qbits"] = [self.auxiliar_qbits_number]
        self.pdf["delta_theta"] = self.delta_theta
        self.pdf["angle_method"] = [self.angle_name]
        self.pdf["angles"] = [self.angles]
        self.pdf["qpu"] = [self.qpu]
        self.pdf["shots"] = [self.shots]
        self.pdf["KS"] = [self.ks]
        self.pdf["fidelity"] = [self.fidelity]
        self.pdf["elapsed_time"] = [self.elapsed_time]
        self.pdf["quantum_time"] = [self.quantum_time[0]]

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
        help="Number of qbits for unitary operator.",
        default=None,
    )
    parser.add_argument(
        "-aux_qbits",
        dest="aux_qbits",
        type=int,
        help="Number of auxiliary qubits for QPE",
        default=None,
    )
    parser.add_argument(
        "-shots",
        dest="shots",
        type=int,
        help="Number of shots: Only valid number is 0 (for exact simulation)",
        default=None,
    )
    parser.add_argument(
        "-angles",
        dest="angles",
        type=int,
        help="Select the angle load method: 0->exact. 1->random",
        default=None,
    )
    #QPU argument
    parser.add_argument(
        "-json_qpu",
        dest="json_qpu",
        type=str,
        default="qpu/qpu.json",
        help="JSON with the qpu configuration",
    )
    parser.add_argument(
        "-id",
        dest="id",
        type=int,
        help="For executing only one element of the list (select the QPU)",
        default=None,
    )
    parser.add_argument(
        "-repetitions",
        dest="repetitions",
        type=int,
        help="Number of repetitions the integral will be computed."+
        "Default: 1",
        default=1,
    )
    parser.add_argument(
        "-folder",
        dest="folder_path",
        type=str,
        help="Path for storing folder",
        default="./",
    )
    parser.add_argument(
        "-name",
        dest="base_name",
        type=str,
        help="Additional name for the generated files",
        default="./",
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
        "--save",
        dest="save",
        default=False,
        action="store_true",
        help="For saving results",
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

    if args.angles == 0:
        angles = 'exact'
    elif args.angles == 1:
        angles = 'random'
    else:
        raise ValueError("angles parameter can be only 0 or 1")

    configuration = {
        "number_of_qbits" : args.n_qbits,
        "auxiliar_qbits_number" : args.aux_qbits,
        "shots" : args.shots,
        "angles" : angles
    }
    if args.count:
        print(len(final_list))
    if args.print:
        if args.id is not None:
            print("##### QPE configuration #####")
            print(configuration)
            print("##### QPU configuration #####")
            print(final_list[args.id])
        else:
            print("##### Posible list of QPUs configuration #####")
            print(final_list)
    if args.execution:
        if args.id is not None:

            for i in range(args.repetitions):
                configuration = {
                    "number_of_qbits" : args.n_qbits,
                    "auxiliar_qbits_number" : args.aux_qbits,
                    "shots" : args.shots,
                    "angles" : angles
                }

                configuration.update({"qpu": select_qpu(final_list[args.id])})
                qpe_rz_b = QPE_RZ(**configuration)
                qpe_rz_b.exe()
                print(qpe_rz_b.pdf)
                print('KS: {}'.format(list(qpe_rz_b.pdf['KS'])[0]))
                print('fidelity: {}'.format(list(qpe_rz_b.pdf['fidelity'])[0]))
                save_folder = args.folder_path
                base_name = args.base_name
                save_name = save_folder + str(args.id) + "_"  + str(base_name) +  ".csv"
                save(args.save, save_name, qpe_rz_b.pdf, "a")

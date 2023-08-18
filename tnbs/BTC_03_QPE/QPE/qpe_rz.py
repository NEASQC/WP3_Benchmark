"""
Class for executing QPE on a R_z^n operator


Author: Gonzal Ferro
"""

import time
import numpy as np
import pandas as pd
from scipy.stats import norm, entropy, chisquare, chi2
import sys
sys.path.append('../')
import QPE.rz_lib as rz_lib

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
            error_text = "Provide the number of auxiliar qubits for QPE"
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

        # Shots for measuring thw QPE circuit
        self.shots = kwargs.get("shots", None)
        if self.shots is not None:
            text = "BE AWARE! The keyword shots should be None because" +\
                "shots should be computed in function of the theorical" +\
                "eigenvalues. You can only provide 0 for doing some testing" +\
                "in the class. 0 will imply complete simnulation of QPE circuit"
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

        # Computing complete time of the procces
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
        help="Number of auxiliar qbits for QPE",
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

    args = parser.parse_args()
    print(args)

    if args.angles == 0:
        angles = 'exact'
    elif args.angles == 1:
        angles = 'random'
    else:
        raise ValueError("angles parameter can be only 0 or 1")

    configuration = {
        "number_of_qbits" : args.n_qbits,
        "auxiliar_qbits_number" : args.aux_qbits,
        "qpu" : rz_lib.get_qpu(args.qpu),
        "shots" : args.shots,
        "angles" : angles
    }
    qpe_rz_b = QPE_RZ(**configuration)
    qpe_rz_b.exe()
    print(qpe_rz_b.pdf)
    print('KS: {}'.format(list(qpe_rz_b.pdf['KS'])[0]))
    print('fidelity: {}'.format(list(qpe_rz_b.pdf['fidelity'])[0]))

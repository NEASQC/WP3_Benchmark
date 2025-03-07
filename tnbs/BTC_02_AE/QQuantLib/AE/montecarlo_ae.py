"""
This module contains the necessary functions and classes to implement a
Monte-Carlo Amplitude Estimation routine. Given a quantum oracle operator,
the amplitude of a given target state is estimated by evaluating it.
Grover-like operators are not used in this routine.


Author: Gonzalo Ferro Costas & Alberto Manzano Herrero

"""

import time
#from copy import deepcopy
import numpy as np
import pandas as pd
import qat.lang.AQASM as qlm
from QQuantLib.utils.data_extracting import get_results
from QQuantLib.utils.utils import check_list_type, measure_state_probability


class MCAE:
    """
    Class for MonteCarlo Amplitude Estimation (MCAE).

    Parameters
    ----------
    oracle: QLM gate
        QLM gate with the Oracle for implementing the
        Grover operator
    target : list of ints
        python list with the target for the amplitude estimation
    index : list of ints
        qubits which mark the register to do the amplitude
        estimation

    kwargs : dictionary
        dictionary that allows the configuration of the IQAE algorithm: \\
        Implemented keys:

    qpu : kwargs, QLM solver
        solver for simulating the resulting circuits
    shots : kwargs, int
        number of measurements
    mcz_qlm : kwargs, bool
        for using or not QLM implementation of the multi controlled Z
        gate
    """

    def __init__(self, oracle: qlm.QRoutine, target: list, index: list, **kwargs):
        """

        Method for initializing the class
        """
        # Setting attributes
        self._oracle = oracle
        self._target = check_list_type(target, int)
        self._index = check_list_type(index, int)

        # Set the QPU to use
        self.linalg_qpu = kwargs.get("qpu", None)
        if self.linalg_qpu is None:
            raise ValueError("Not QPU was provide. Please provide it!")

        self.shots = int(kwargs.get("shots", 100))
        self.mcz_qlm = kwargs.get("mcz_qlm", True)

        self.ae_l = None
        self.ae_u = None
        self.theta_l = None
        self.theta_u = None
        self.theta = None
        self.ae = None
        self.circuit_statistics = None
        self.time_pdf = None
        self.run_time = None
        self.schedule = {}
        self.oracle_calls = None
        self.max_oracle_depth = None
        self.schedule_pdf = None
        self.quantum_times = []
        self.quantum_time = None

    #####################################################################
    @property
    def oracle(self):
        """
        creating oracle property
        """
        return self._oracle

    @oracle.setter
    def oracle(self, value):
        """
        setter of the oracle property
        """
        self._oracle = value

    @property
    def target(self):
        """
        creating target property
        """
        return self._target

    @target.setter
    def target(self, value):
        """
        setter of the target property
        """
        self._target = check_list_type(value, int)

    @property
    def index(self):
        """
        creating index property
        """
        return self._index

    @index.setter
    def index(self, value):
        """
        setter of the index property
        """
        self._index = check_list_type(value, int)

    #####################################################################

    def run(self):
        r"""
        run method for the class.

        Returns
        ----------

        self.ae :
            amplitude estimation parameter

        """

        #Done Measurements on the oracle
        start = time.time()
        results, circuit, _, _ = get_results(
            self.oracle,
            linalg_qpu=self.linalg_qpu,
            shots=self.shots,
            qubits=self.index
        )
        end = time.time()
        self.quantum_times.append(end-start)
        self.ae = measure_state_probability(results, self.target)
        self.ae_l = max(0.0, self.ae - 2.0 / np.sqrt(float(self.shots)))
        self.ae_u = min(1.0, self.ae + 2.0 / np.sqrt(float(self.shots)))
        self.run_time = end - start
        self.schedule_pdf = pd.DataFrame(
            [[0, self.shots]],
            columns=['m_k', 'shots']
        )
        self.oracle_calls = np.sum(
            self.schedule_pdf['shots'] * (2 * self.schedule_pdf['m_k'] + 1))
        self.max_oracle_depth = np.max(2 *  self.schedule_pdf['m_k']+ 1)
        self.quantum_time = sum(self.quantum_times)
        return self.ae

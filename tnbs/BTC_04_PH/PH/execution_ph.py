"""
For executing a VQE quantum step of a ansatz and a given
Parent Hamiltonian
Author: Gonzalo Ferro
"""

import logging
import ast
import time
import pandas as pd
import numpy as np
from qat.core import Observable, Term

logging.basicConfig(
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    #level=logging.INFO
    level=logging.DEBUG
)
logger = logging.getLogger("__name__")

class PH_EXE:
    """
    Class for, given an ansatz and its parent hamiltonian Pauli
    decompostion, execute a VQE step

    Parameters
    ----------

    ansatz : QLM circuit
        QLM circuit with the input ansatz
    n_qubits : int
        number of qubits of the ansatz
    pauli_ph : pandas DataFrame
        pauli decomposition of the parent Hamiltonian
    kwars : dictionary
        For configuring the class

    """
    def __init__(self, ansatz, pauli_ph, nqubits, **kwargs):
        """

        Method for initializing the class
        """

        self.ansatz = ansatz
        self.pauli_ph = pauli_ph
        self.nqubits = nqubits
        self.kwargs = kwargs
        self.pdf_info = pd.DataFrame(self.kwargs, index=[0])
        # For Saving
        self._save = kwargs.get("save", False)
        self.filename = kwargs.get("filename", None)
        self.qpu = kwargs.get("qpu", None)
        self.nb_shots = kwargs.get("nb_shots", None)
        if self.nb_shots is None:
            self.nb_shots = 0
        self.truncation = kwargs.get("truncation", None)
        if self.truncation is not None:
            self.truncation = 10 ** (-self.truncation)

        self.pauli_pdf = None
        self.observable_time = None
        self.quantum_time = None
        self.gse = None
        self.pdf_result = None

    def run(self):
        """
        Execute VQE step
        """
        if self.truncation is not None:
            index = abs(self.pauli_ph["PauliCoefficients"]) > self.truncation
            self.pauli_pdf = self.pauli_ph[index]
            logger.debug("Additional truncation of Pauli Coeficients")
        else:
            self.pauli_pdf = self.pauli_ph
        pauli_coefs = list(self.pauli_pdf["PauliCoefficients"])
        pauli_strings = list(self.pauli_pdf["PauliStrings"])
        affected_qubits = list(self.pauli_pdf["Qbits"])

        logger.info("Creating Observables for Pauli configuration")
        tick = time.time()
        terms = [Term(coef, ps, qb) for coef, ps, qb in \
            zip(pauli_coefs, pauli_strings, affected_qubits)]

        # Creates the atos myqlm observable
        observable = Observable(nqbits=self.nqubits, pauli_terms=terms)
        tack = time.time()
        self.observable_time = tack - tick
        logger.info("Observable time: %f", self.observable_time)

        # Creates the job
        job_ansatz = self.ansatz.to_job(
            'OBS', observable=observable, nbshots=self.nb_shots)

        logger.info('Execution Local Parent Hamiltonian')
        tick_q = time.time()
        gse = self.qpu.submit(job_ansatz)
        self.gse = gse.value
        tock = time.time()
        self.quantum_time = tock - tick_q
        logger.info("GSE: %f", self.gse)
        text = [
            "gse", "observable_time", "quantum_time", "n_coefs",
            "n_trunc_coefs"]
        values = [
            self.gse, self.observable_time, self.quantum_time,
            len(self.pauli_ph), len(self.pauli_pdf)]
        self.pdf_result = pd.DataFrame(
            values,
            index=text
        ).T
        if self._save:
            self.save()
    def save(self):
        """
        Saving Staff
        """
        pdf = pd.concat([self.pdf_info, self.pdf_result], axis=1)
        pdf.to_csv(
            self.filename+"_phexe.csv", sep=";")

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
        self.t_inv = kwargs.get("t_inv", None)

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

        logger.debug("Creating Observables for Pauli configuration")
        tick = time.time()
        terms = [Term(coef, ps, qb) for coef, ps, qb in \
            zip(pauli_coefs, pauli_strings, affected_qubits)]
        # Creates the atos myqlm observable
        observable = Observable(nqbits=self.nqubits, pauli_terms=terms)
        tack = time.time()
        self.observable_time = tack - tick
        logger.debug("Observable time: %f", self.observable_time)

        # Creates the job
        job_ansatz = self.ansatz.to_job(
            'OBS', observable=observable, nbshots=self.nb_shots)

        logger.debug('Execution Local Parent Hamiltonian')
        tick_q = time.time()
        gse = self.qpu.submit(job_ansatz)
        self.gse = gse.value
        tock = time.time()
        self.quantum_time = tock - tick_q
        logger.debug("GSE: %f", self.gse)
        text = [
            "observable_time", "quantum_time", "n_coefs", "n_trunc_coefs", "gse"
        ]
        values = [
            self.observable_time, self.quantum_time,
            len(self.pauli_ph), len(self.pauli_pdf), self.gse
        ]

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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nb_shots",
        dest="nb_shots",
        type=int,
        help="Number of shots",
        default=0,
    )
    parser.add_argument(
        "-qpu_ph",
        dest="qpu_ph",
        type=str,
        default=None,
        help="QPU for parent hamiltonian simulation: [qlmass, python, c]",
    )
    parser.add_argument(
        "-truncation",
        dest="truncation",
        type=int,
        help="Truncation for Pauli coeficients.",
        default=None,
    )
    parser.add_argument(
        "--t_inv",
        dest="t_inv",
        default=False,
        action="store_true",
        help="Setting translational invariant of the ansatz",
    )
    parser.add_argument(
        "--save",
        dest="save",
        default=False,
        action="store_true",
        help="For storing results",
    )
    #Execution argument
    parser.add_argument(
        "--exe",
        dest="execution",
        default=False,
        action="store_true",
        help="For executing program",
    )
    args = parser.parse_args()

    # Given ansatz with parameters and Pauli of PH executes VQE step
    from utils import get_filelist, get_info_basefn, get_qpu
    from ansatzes import ansatz_selector



    folder = "/mnt/netapp1/Store_CESGA/home/cesga/gferro/PH/"
    folder_n = "kk/"
    depth, nqubits, ansatz = get_info_basefn(folder_n)
    # Read the state from csv
    base_fn = get_filelist(folder + folder_n)[0]

    # Ansatz Configuration
    ansatz_conf = {
        'nqubits' : nqubits,
        'depth' : depth
    }

    # Create Ansatz Circuit
    logger.info("Creating ansatz circuit")
    circuit = ansatz_selector(ansatz, **ansatz_conf)
    # Load Parameters
    logger.info("Loading Parameters")
    parameters_pdf = pd.read_csv(
        base_fn + "_parameters.csv", sep=";", index_col=0)
    # Formating Parameters
    parameters = {k:v for k, v in zip(
        parameters_pdf['key'], parameters_pdf['value'])}
    circuit = circuit(**parameters)

    # Loading PH Pauli decomposition
    logger.info("Loading PH Pauli decomposition")

    # Loading Pauli
    pauli_pdf = pd.read_csv(
        base_fn + "_pauli.csv", sep=";", index_col=0)
    affected_qubits = [ast.literal_eval(i_) for i_ in list(pauli_pdf["Qbits"])]
    pauli_pdf["Qbits"] = affected_qubits

    # Executing VQE step
    logger.info("Executing VQE step")
    vqe_conf = {
        "qpu" : get_qpu(args.qpu_ph),
        "nb_shots": args.nb_shots,
        "truncation": args.truncation,
        "filename": base_fn,
        "save": args.save
    }
    exe_ph = PH_EXE(circuit, pauli_pdf, nqubits, **vqe_conf)
    if args.execution:
        exe_ph.run()

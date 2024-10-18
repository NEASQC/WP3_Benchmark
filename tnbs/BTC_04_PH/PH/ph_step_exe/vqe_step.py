"""
For executing a VQE quantum step of a ansatz and a given
Parent Hamiltonian.

Author: Gonzalo Ferro
"""

import sys
import logging
import time
import re
import ast
import pandas as pd
from qat.core import Observable, Term
sys.path.append("../../")
from PH.ansatzes.ansatzes import ansatz_selector, angles_ansatz01
logger = logging.getLogger('__name__')


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
        self.pdf = None

    def run(self):
        """
        Execute VQE step
        """
        if self.truncation is not None:
            index = abs(self.pauli_ph["PauliCoefficients"]) > self.truncation
            self.pauli_pdf = self.pauli_ph[index]
            logger.info(
                "Additional truncation of Pauli Coeficients: {}".format(
                len(self.pauli_pdf)))
        else:
            self.pauli_pdf = self.pauli_ph
        pauli_coefs = list(self.pauli_pdf["PauliCoefficients"])
        pauli_strings = list(self.pauli_pdf["PauliStrings"])
        affected_qubits = list(self.pauli_pdf["Qbits"])

        if self.t_inv:
            # For translational invariant ansatzes we must replicate
            # the pauli terms for all the qubits
            print("Invariant Ansatz: Replicating Pauli Strings accross qubits")
            logger.info(
                "Invariant Ansatz: Replicating Pauli Strings accross qubits")
            terms = len(pauli_coefs)
            pauli_coefs = pauli_coefs * self.nqubits
            pauli_strings = pauli_strings * self.nqubits
            qubits_list = []
            for qb_pos in range(self.nqubits):
                step = [(qb_pos + k) % self.nqubits for k in range(
                    len(affected_qubits[0]))]
                qubits_list = qubits_list + [step] * terms
            affected_qubits = qubits_list

        logger.info("Creating Observables for Pauli configuration")
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
        self.pdf = pd.concat([self.pdf_info, self.pdf_result], axis=1)
        if self._save:
            self.save()

    def save(self):
        """
        Saving Staff
        """
        self.pdf.to_csv(
            self.filename+"_phexe.csv", sep=";")

#def get_info_basefn(base_fn):
#    depth = int(re.findall(r"depth_(.*)_qpu", base_fn)[0])
#    nqubits = int(re.findall(r"nqubits_(.*)_depth_", base_fn)[0])
#    ansatz = re.findall(r"ansatz_(.*)_nqubits", base_fn)[0]
#    return depth, nqubits, ansatz
def get_info_basefn(base_fn):
    nqubits = int(re.findall(r"nqubits_(.*)_depth_", base_fn)[0])
    ansatz = re.findall(r"ansatz_(.*)_nqubits", base_fn)
    if len(ansatz) == 0:
        ansatz = "simple01"
    else:
        ansatz = ansatz[0]
    depth = re.findall(r"depth_(.*)_qpu", base_fn)
    if len(depth) == 0:
        depth = re.findall(r"depth_(\d*)", base_fn)[0]
    else:
        depth = depth[0]
    depth = int(depth)
    return depth, nqubits, ansatz

def run_ph_execution(**configuration):
    """
    Given an ansatz circuit, the parameters and the Pauli decomposition
    of the corresponding local PH executes a VQE step for computing
    the energy of the ansatz under the Hamiltonian that MUST BE near 0
    Given an input base_fn that MUST have following pattern:
        * base_fn = ansatz_{}_nqubits_{}_depth_{}_qpu_ansatz_{}
    Additionally folowing files MUST exist:
        * {base_fn}_parameters.csv
        * {base_fn}_pauli.csv
    The functions gets the information about: ansatz, nqubits and depth
    and executes the following Workflow:
        1. Create QLM circuit using the ansatz type readed from the folder
        2 Loading parameters for the circuit from: {}_parameters.csv
        3. Loading Pauli Decomposition from: {}_pauli.csv
        4. Executes VQE step.
    If save is True the result of the execution is stored as:
        * {base_fn}_phexe.csv
    """

    # 1. Create QLM circuit using the ansatz type readed from the folder
    logger.info("Creating ansatz circuit")
    base_fn = configuration["base_fn"]
    depth, nqubits, ansatz = get_info_basefn(base_fn)
    text = "ansatz: {0}, nqubits: {1} depth: {2}".format(ansatz, nqubits, depth)
    logger.debug(text)
    ansatz_conf = {
        "nqubits" :nqubits,
        "depth" : depth,
    }
    circuit = ansatz_selector(ansatz, **ansatz_conf)

    # 2 Loading parameters for the circuit from: {}_parameters.csv
    text = "Loading Parameters from: {}".format(base_fn + "_parameters.csv")
    logger.info(text)
    parameters_pdf = pd.read_csv(
        base_fn + "_parameters.csv", sep=";", index_col=0)
    # Formating Parameters
    circuit, _ = angles_ansatz01(circuit, parameters_pdf)
    #from qat.core.console import display
    #display(circuit)

    # 3. Loading Pauli Decomposition from: {}_pauli.csv
    text = "Loading PH Pauli decomposition from: {}".format(
        base_fn + "_parameters.csv")
    logger.info(text)
    # Loading Pauli
    pauli_pdf = pd.read_csv(
        base_fn + "_pauli.csv", sep=";", index_col=0)
    affected_qubits = [ast.literal_eval(i_) for i_ in list(pauli_pdf["Qbits"])]
    pauli_pdf["Qbits"] = affected_qubits

    # 4. Executes VQE step.
    logger.info("Executing VQE step")
    vqe_conf = {
        "qpu" : configuration["qpu"],
        "nb_shots": configuration["nb_shots"],
        "truncation": configuration["truncation"],
        "t_inv": configuration["t_inv"],
        "filename": base_fn,
        "save": configuration["save"],
    }
    exe_ph = PH_EXE(circuit, pauli_pdf, nqubits, **vqe_conf)
    exe_ph.run()
    return exe_ph.pdf

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s-%(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO
        #level=logging.DEBUG
    )
    logger = logging.getLogger('__name__')
    import argparse
    import json
    sys.path.append("../../../")
    from qpu.select_qpu import select_qpu
    from qpu.benchmark_utils import combination_for_list

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-basefn",
        dest="base_fn",
        type=str,
        default="",
        help="Base Filename for Loading Files",
    )
    parser.add_argument(
        "-nb_shots",
        dest="nb_shots",
        type=int,
        help="Number of shots",
        default=0,
    )
    parser.add_argument(
        "-truncation",
        dest="truncation",
        type=int,
        help="Truncation for Pauli coeficients.",
        default=None,
    )
    parser.add_argument(
        "-json_qpu",
        dest="json_qpu",
        type=str,
        default="../qpu/qpu_ideal.json",
        help="JSON with the qpu configuration",
    )
    parser.add_argument(
        "-qpu_id",
        dest="qpu_id",
        type=int,
        default=None,
        help="Select a QPU from a JSON file",
    )
    parser.add_argument(
        "--t_inv",
        dest="t_inv",
        default=False,
        action="store_true",
        help="Setting translational invariant of the ansatz",
    )
    parser.add_argument(
        "--print",
        dest="print",
        default=False,
        action="store_true",
        help="For printing the selected QPU configuration."
    )
    #Execution argument
    parser.add_argument(
        "--exe",
        dest="execution",
        default=False,
        action="store_true",
        help="For executing program",
    )
    parser.add_argument(
        "--save",
        dest="save",
        default=False,
        action="store_true",
        help="For storing results",
    )
    args = parser.parse_args()
    with open(args.json_qpu) as json_file:
        qpu_cfg = json.load(json_file)
    final_list = combination_for_list(qpu_cfg)
    if args.print:
        if args.qpu_id is not None:
            print(final_list[args.qpu_id])
        else:
            print("All possible QPUS will be printed: ")
            print(final_list)
    if args.execution:
        if args.qpu_id is not None:
            qpu_config = final_list[args.qpu_id]
            config = vars(args)
            config.update({"qpu": select_qpu(qpu_config)})
            print(run_ph_execution(**config))
        else:
            raise ValueError(
                "BE AWARE. For execution the -qpu_id is Mandatory"
            )

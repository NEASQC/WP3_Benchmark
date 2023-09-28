"""
For retrieving results from QLM
Author: Gonzalo Ferro
"""

import panda as pd
from datetime import datetime
from qat.qlmaas import QLMaaSConnection
from ansatzes import proccess_qresults

def get_info(job, base_fn, save):
    """
    Given a QLM job getting the correspondient information

    parameters
    ----------

    jobid : string
        JobId of the job for retrieval
    base_fn : string
        base file name for saving the state
    save : bool
        For saving the state of the ansatz

    returns
    _______

    nqbits : int
        number of qubits of the circuit
    elapsed : float
        seconds for solving the ansatz
    pdf : pandas DataFrame
        DataFrame with the results
    """
    # Open QLM connection
    connection = QLMaaSConnection()
    # Get Info of the job
    job_info = connection.get_job_info(job)
    nbqbits = job_info.resources.nbqbits
    end = datetime.strptime(
        job_info.ending_date.rsplit(".")[0],
        "%Y-%m-%d %H:%M:%S")
    start = datetime.strptime(
        job_info.starting_date.rsplit(".")[0],
        "%Y-%m-%d %H:%M:%S")
    elapsed = end - start
    elapsed = elapsed.total_seconds()
    pdf = proccess_qresults(connection.get_result(job), nbqbits)
    solve_ansatz_time = pd.DataFrame([elapsed], index=["solve_ansatz_time"]).T
    # Saving the state
    if save:
        pdf.to_csv(base_fn+"_state.csv", sep=";")
        solve_ansatz_time.to_csv(base_fn+"_solve_ansatz_time.csv", sep=";")
    return nbqbits, elapsed, pdf

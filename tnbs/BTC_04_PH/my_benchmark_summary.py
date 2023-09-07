"""
Template for properly formating the results of a benchamark kernel for
filling Results sub-field of Benchmarks field
"""

from collections import OrderedDict
import psutil

def summarize_results(**kwargs):
    """
    Mandatory code for properly present the benchmark results following
    the NEASQC jsonschema
    """

    # n_qbits = [4]
    # #Info with the benchmark results like a csv or a DataFrame
    # pdf = None
    # #Metrics needed for reporting. Depend on the benchmark kernel
    # list_of_metrics = ["MRSE"]

    import pandas as pd
    benchmark_file = kwargs.get("benchmark_file", None)
    index_columns = [0, 1, 2, 3, 4, 5]
    pdf = pd.read_csv(benchmark_file, header=[0, 1], index_col=index_columns)
    pdf.reset_index(inplace=True)
    n_qbits = list(set(pdf["nqubits"]))
    depth = list(set(pdf["depth"]))
    list_of_metrics = ["gse"]

    results = []
    #If several qbits are tested
    # For ordering by n_qbits
    for n_ in n_qbits:
        for depth_ in depth:
            # For ordering by auxiliar qbits
            result = OrderedDict()
            result["NumberOfQubits"] = n_
            result["QubitPlacement"] = list(range(n_))
            result["QPUs"] = [1]
            result["CPUs"] = psutil.Process().cpu_affinity()
            #Select the proper data
            indice = (pdf["nqubits"] == n_) & (pdf["depth"] == depth_)
            step_pdf = pdf[indice]
            result["TotalTime"] = step_pdf["elapsed_time"]["mean"].iloc[0]
            result["SigmaTotalTime"] = step_pdf["elapsed_time"]["std"].iloc[0]
            result["QuantumTime"] = step_pdf["quantum_time"]["mean"].iloc[0]
            result["SigmaQuantumTime"] = step_pdf["quantum_time"]["std"].iloc[0]
            result["ClassicalTime"] = step_pdf["classic_time"]["mean"].iloc[0]
            result["SigmaClassicalTime"] = step_pdf["classic_time"]["std"].iloc[0]

            # For identifying the test
            # result['AnsatzName'] = step_pdf["ansatz"].iloc[0]
            result['AnsatzDepth'] = depth_
            # result['QPUforAnsatz'] = step_pdf["qpu_ansatz"].iloc[0]
            # result['QPUforPH'] = step_pdf["qpu_ph"].iloc[0]
            result['Shots'] = int(step_pdf['nb_shots'].iloc[0])
            metrics = []
            #For each fixed number of qbits several metrics can be reported
            for metric_name in list_of_metrics:
                metric = OrderedDict()
                #MANDATORY
                metric["Metric"] = metric_name
                metric["Value"] = step_pdf[metric_name]["mean"].iloc[0]
                metric["STD"] = step_pdf[metric_name]["std"].iloc[0]
                metric["COUNT"] = int(step_pdf[metric_name]["count"].iloc[0])
                metrics.append(metric)
            result["Metrics"] = metrics
            results.append(result)
    return results

if __name__ == "__main__":
    import json
    import jsonschema
    json_file = open(
        "../../tnbs/templates/NEASQC.Benchmark.V2.Schema_modified.json"
    )
    schema = json.load(json_file)
    json_file.close()

    ################## Configuring the files ##########################

    configuration = {
        "benchmark_file" : "Results/kernel_SummaryResults.csv"
    }

    ######## Execute Validations #####################################


    schema_bench = schema['properties']['Benchmarks']['items']['properties']
    print("Validate Results")
    print(summarize_results(**configuration))
    try:
        jsonschema.validate(
            instance=summarize_results(**configuration),
            schema=schema_bench['Results']
        )
        print("\t Results is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

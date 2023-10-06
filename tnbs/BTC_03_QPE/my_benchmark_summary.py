"""
This module is for properly formating the Results sub-field of
Benchmarks field of the NEASQC JSON report by gathering the results of
a complete BTC of the QPE kernel.
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
    n_qbits = list(set(pdf["n_qbits"]))
    angle_methods = list(set(pdf["angle_method"]))
    aux_qbits = list(set(pdf["aux_qbits"]))
    list_of_metrics = [
        "KS", "fidelity",
    ]

    results = []
    #If several qbits are tested
    # For ordering by n_qbits
    for n_ in n_qbits:
        # For ordering by auxiliar qbits
        for aux_ in aux_qbits:
            for angle_ in angle_methods:
                result = OrderedDict()
                result["NumberOfQubits"] = n_
                result["QubitPlacement"] = list(range(n_))
                result["QPUs"] = [1]
                result["CPUs"] = psutil.Process().cpu_affinity()
                #Select the proper data
                indice = (pdf['n_qbits'] == n_) & (pdf['aux_qbits'] == aux_) \
                    & (pdf['angle_method'] == angle_)
                step_pdf = pdf[indice]
                result["TotalTime"] = step_pdf["elapsed_time"]["mean"].iloc[0]
                result["SigmaTotalTime"] = step_pdf["elapsed_time"]["std"].iloc[0]
                result["QuantumTime"] = step_pdf["quantum_time"]["mean"].iloc[0]
                result["SigmaQuantumTime"] = step_pdf["quantum_time"]["std"].iloc[0]
                result["ClassicalTime"] = step_pdf["classic_time"]["mean"].iloc[0]
                result["SigmaClassicalTime"] = step_pdf["classic_time"]["std"].iloc[0]

                # For identifying the test
                result['AuxiliarNumberOfQubits'] = aux_
                result['MethodForSettingAngles'] = angle_
                result['QPEAnglePrecision'] = float(step_pdf['delta_theta'].iloc[0])
                result['Shots'] = int(step_pdf['shots'].iloc[0])
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

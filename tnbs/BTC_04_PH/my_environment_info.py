"""
Module for gathering different hardware information for main fields of
NEASQC report generation for reporting BTC of a PL kernel
"""
import platform
import psutil
import json
from collections import OrderedDict

import sys
l_sys = sys.path
l_path = l_sys[['BTC_01' in i for i in l_sys].index(True)]
sys.path.append(l_path+'/PL')

def my_ideal_qpu(**kwargs):
    """
    Define an ideal qpu
    """
    from qat.lang.AQASM import gates as qlm_gates
    #Defining the Qubits of the QPU
    qubits = OrderedDict()
    qubits["QubitNumber"] = 0
    qubits["T1"] = 1.0
    qubits["T2"] = 1.0

    #Defining the Gates of the QPU
    gates = OrderedDict()
    gates["Gate"] = "none"
    gates["Type"] = "Single"
    gates["Symmetric"] = False
    gates["Qubits"] = [0]
    gates["MaxTime"] = 1.0


    #Defining the Basic Gates of the QPU
    qpus = OrderedDict()
    qpus["BasicGates"] = [gate for gate in list(qlm_gates.default_gate_set())]
    qpus["Qubits"] = [qubits]
    qpus["Gates"] = [gates]
    qpus["Technology"] = "other"

    qpu_description = OrderedDict()
    qpu_description['NumberOfQPUs'] = 1
    qpu_description['QPUs'] = [qpus]
    return qpu_description

def my_noisy_qpu(qpu_config):
    """
    Define an ideal qpu
    """
    from PL.qpu.select_qpu import select_qpu
    my_noisy_qpu = select_qpu(qpu_config)
    time_gate_dict = my_noisy_qpu.hardware_model.gates_specification.gate_times
    gate_list = []
    for gate_, value_ in my_noisy_qpu.hardware_model.gate_noise.items():
        if value_.keywords["nqbits"] in [1, 2]:
            gate_step = OrderedDict()
            gate_step["Gate"] = gate_
            if value_.keywords["nqbits"] == 1:
                gate_step["Type"] = "Single"
            if value_.keywords["nqbits"] == 2:
                gate_step["Type"] = "Entanglement"
            gate_step["Symmetric"] = False
            gate_step["Qubits"] = [0]
            if time_gate_dict[gate_] is None:
                gate_step["MaxTime"] = 1.0
            else:
                if time_gate_dict[gate_] == 0:
                    gate_step["MaxTime"] = 1.0
                else:
                    gate_step["MaxTime"] = time_gate_dict[gate_]
            gate_step["Error"] = value_.keywords["rb_eps"]
            gate_list.append(gate_step)
        
    qubits = OrderedDict()
    qubits["QubitNumber"] = 0
    if qpu_config["idle"]["amplitude_damping"] == True:
        qubits["T1"] = qpu_config["idle"]["t1"]
    else:
         qubits["T1"] = 1.0
    if qpu_config["idle"]["dephasing_channel"] == True:
        qubits["T2"] = qpu_config["idle"]["t2"]
    else:
         qubits["T2"] = 1.0
        
    #Defining the Basic Gates of the QPU
    qpus = OrderedDict()
    qpus["BasicGates"] = [g_["Gate"] for g_ in gate_list]
    qpus["Qubits"] = [qubits]
    qpus["Gates"] = gate_list
    qpus["Technology"] = "other"
    qpu_description = OrderedDict()
    qpu_description['NumberOfQPUs'] = 1
    qpu_description['QPUs'] = [qpus]
    return qpu_description


def my_organisation(**kwargs):
    """
    Given information about the organisation how uploads the benchmark
    """
    #name = "None"
    name = "CESGA"
    return name

def my_machine_name(**kwargs):
    """
    Name of the machine where the benchmark was performed
    """
    #machine_name = "None"
    machine_name = platform.node()
    return machine_name

def my_qpu_model(**kwargs):
    """
    Name of the model of the QPU
    """
    json_file = kwargs.get("qpu")
    with open(json_file, 'r') as openfile:
        #Reading from json file
        json_object = json.load(openfile)
    #qpu_model = "None"
    qpu_model = json_object["qpu_type"]
    return qpu_model

def my_qpu(**kwargs):
    """
    Complete info about the used QPU
    """
    #Basic schema
    #QPUDescription = {
    #    "NumberOfQPUs": 1,
    #    "QPUs": [
    #        {
    #            "BasicGates": ["none", "none1"],
    #            "Qubits": [
    #                {
    #                    "QubitNumber": 0,
    #                    "T1": 1.0,
    #                    "T2": 1.00
    #                }
    #            ],
    #            "Gates": [
    #                {
    #                    "Gate": "none",
    #                    "Type": "Single",
    #                    "Symmetric": False,
    #                    "Qubits": [0],
    #                    "MaxTime": 1.0
    #                }
    #            ],
    #            "Technology": "other"
    #        },
    #    ]
    #}

    json_file = kwargs.get("qpu")
    with open(json_file, 'r') as openfile:
        #Reading from json file
        json_object = json.load(openfile)
    qpu_model = json_object["qpu_type"]
    ideal_qpus = ["c", "python", "linalg", "mps", "qlmass_linalg", "qlmass_mps"]
    if qpu_model in ideal_qpus:
        return my_ideal_qpu(**kwargs)
    if qpu_model in ["ideal", "noisy"]:
        return my_noisy_qpu(json_object)
        
        

    # #Defining the Qubits of the QPU
    # qubits = OrderedDict()
    # qubits["QubitNumber"] = 0
    # qubits["T1"] = 1.0
    # qubits["T2"] = 1.0

    # #Defining the Gates of the QPU
    # gates = OrderedDict()
    # gates["Gate"] = "none"
    # gates["Type"] = "Single"
    # gates["Symmetric"] = False
    # gates["Qubits"] = [0]
    # gates["MaxTime"] = 1.0


    # #Defining the Basic Gates of the QPU
    # qpus = OrderedDict()
    # qpus["BasicGates"] = ["none", "none1"]
    # qpus["Qubits"] = [qubits]
    # qpus["Gates"] = [gates]
    # qpus["Technology"] = "other"

    # qpu_description = OrderedDict()
    # qpu_description['NumberOfQPUs'] = 1
    # qpu_description['QPUs'] = [qpus]

    # return qpu_description

def my_cpu_model(**kwargs):
    """
    model of the cpu used in the benchmark
    """
    #cpu_model = "None"
    cpu_model = platform.processor()
    return cpu_model

def my_frecuency(**kwargs):
    """
    Frcuency of the used CPU
    """
    #Use the nominal frequency. Here, it collects the maximum frequency
    #print(psutil.cpu_freq())
    #cpu_frec = 0
    cpu_frec = psutil.cpu_freq().max/1000
    return cpu_frec

def my_network(**kwargs):
    """
    Network connections if several QPUs are used
    """
    network = OrderedDict()
    network["Model"] = "None"
    network["Version"] = "None"
    network["Topology"] = "None"
    return network

def my_QPUCPUConnection(**kwargs):
    """
    Connection between the QPU and the CPU used in the benchmark
    """
    #
    # Provide the information about how the QPU is connected to the CPU
    #
    qpuccpu_conn = OrderedDict()
    qpuccpu_conn["Type"] = "memory"
    qpuccpu_conn["Version"] = "None"
    return qpuccpu_conn

if __name__ == "__main__":
    """
    For comparing the results of the environment info vs the jsonschema
    """
    import jsonschema
    json_file = open(
        "../../tnbs/templates/NEASQC.Benchmark.V2.Schema_modified.json"
    )

    schema = json.load(json_file)
    json_file.close()

    ################## Configuring the files ##########################

    configuration = {
        "qpu" : "Results/qpu_configuration.json"
    }
    ######## Execute Validations #####################################

    schema_org = schema['properties']['ReportOrganization']
    print("Validate ReportOrganization")
    print(my_organisation(**configuration))
    try:
        jsonschema.validate(
            instance=my_organisation(**configuration),
            schema=schema_org
        )
        print("\tReportOrganization is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    schema_nodename = schema['properties']['MachineName']
    print("Validate MachineName")
    print(my_machine_name(**configuration))
    try:
        jsonschema.validate(
            instance=my_machine_name(**configuration),
            schema=schema_nodename
        )
        print("\tMachineName is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    schema_model = schema['properties']['QPUModel']
    print("Validate QPUModel")
    print(my_qpu_model(**configuration))
    try:
        jsonschema.validate(
            instance=my_qpu_model(**configuration),
            schema=schema_nodename
        )
        print("\tQPUModel is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)


    schema_qpu = schema['properties']['QPUDescription']['items']
    print("Validate QPUDescription")
    print(my_qpu(**configuration))
    try:
        jsonschema.validate(
            instance=my_qpu(**configuration),
            schema=schema_qpu
        )
        print("\tQPUDescription is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    schema_cpu = schema['properties']['CPUModel']
    print("Validate QCPUModel")
    print(my_cpu_model(**configuration))
    try:
        jsonschema.validate(
            instance=my_cpu_model(**configuration),
            schema=schema_cpu
        )
        print("\tQCPUModel is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    schema_frecuency = schema['properties']['Frequency']
    print("Validate Frequency")
    print(my_frecuency(**configuration))
    try:
        jsonschema.validate(
            instance=my_frecuency(**configuration),
            schema=schema_frecuency
        )
        print("\tFrequency is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    schema_network = schema['properties']['Network']
    print("Validate Network")
    print(my_network())
    try:
        jsonschema.validate(
            instance=my_network(**configuration),
            schema=schema_network
        )
        print("\tNetwork is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    schema_qpucpu_conn = schema['properties']['QPUCPUConnection']
    print("Validate QPUCPUConnection")
    print(my_QPUCPUConnection(**configuration))
    try:
        jsonschema.validate(
            instance=my_QPUCPUConnection(**configuration),
            schema=schema_qpucpu_conn
        )
        print("\tQPUCPUConnection is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)



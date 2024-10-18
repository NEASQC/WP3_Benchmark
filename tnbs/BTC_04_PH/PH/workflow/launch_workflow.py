"""
For launching a complete workflow PH execution
Author: Gonzalo Ferro
"""

import sys
import json
sys.path.append("../../")
from PH.utils.utils_ph import combination_for_list
from PH.utils.utils_ph import cartesian_product
from PH.workflow.workflow import workflow
sys.path.append("../../../")
from qpu.select_qpu import select_qpu

def run_id(**configuration):
    qpu_config = {"qpu_type": configuration["qpu_ansatz"]}
    configuration.update({"ansatz_qpu": select_qpu(qpu_config)})
    configuration.update({"ph_qpu": select_qpu(configuration)})
    #print(configuration["ph_qpu"])
    pdf = workflow(**configuration)
    return pdf




if __name__ == "__main__":
    import logging
    logging.basicConfig(
        format='%(asctime)s-%(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO
        #level=logging.DEBUG
    )
    logger = logging.getLogger('__name__')
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        dest="all",
        default=False,
        action="store_true",
        help="For executing complete list",
    )
    group.add_argument(
        "-id",
        dest="id",
        type=int,
        help="For executing only one element of the list",
        default=None,
    )
    parser.add_argument(
        "-qpu_ph",
        dest="qpu_ph",
        type=str,
        default="../qpu/qpu_ideal.json",
        help="JSON with the qpu configuration for ground state computation",
    )
    parser.add_argument(
        "--print",
        dest="print",
        default=False,
        action="store_true",
        help="For printing the AE algorihtm configuration."
    )
    #Execution argument
    parser.add_argument(
        "--exe",
        dest="execution",
        default=False,
        action="store_true",
        help="For executing program",
    )
    #Execution argument
    parser.add_argument(
        "--count",
        dest="count",
        default=False,
        action="store_true",
        help="Getting the number of elements",
    )
    args = parser.parse_args()

    # Creating Combination list
    # Load json file
    json_file = "workflow.json"
    f_ = open(json_file)
    conf = json.load(f_)
    combination_list = combination_for_list(conf)

    # Creating Combination list for QPUs
    with open(args.qpu_ph) as json_file:
        qpu_cfg = json.load(json_file)
    qpu_list = combination_for_list(qpu_cfg)
    # Cartesian product of combination_list and qpu_list
    final_list = cartesian_product(combination_list, qpu_list)

    if args.print:
        if args.id is not None:
            print(final_list[args.id])
        elif args.all:
            print(final_list)
        else:
            print("Provide -id or --all")
    if args.count:
        print("Number of elements: {}".format(len(final_list)))
    if args.execution:
        if args.id is not None:
            configuration = final_list[args.id]
            print(run_id(**configuration))
        if args.all:
            for configuration in final_list:
                print(run_id(**configuration))

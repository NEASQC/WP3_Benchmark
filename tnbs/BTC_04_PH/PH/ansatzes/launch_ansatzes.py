"""
For launching a VQE quantum step execution
Author: Gonzalo Ferro
"""

import sys
import json
sys.path.append("../../")
from PH.ansatzes.ansatzes import run_ansatz
sys.path.append("../../../")
from qpu.select_qpu import select_qpu
from qpu.benchmark_utils import combination_for_list

def run_id(**configuration):
    qpu_config = {"qpu_type": configuration["qpu_ansatz"]}
    print(qpu_config)
    print(select_qpu(qpu_config))
    configuration.update({"qpu": select_qpu(qpu_config)})
    _ = run_ansatz(**configuration)


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
        help="For executing or submitting all anstazes from ansatzes.json",
    )
    group.add_argument(
        "-id",
        dest="id",
        type=int,
        help="Select one element of all the anstazes from ansatzes.json",
        default=None,
    )
    parser.add_argument(
        "--print",
        dest="print",
        default=False,
        action="store_true",
        help="For printing the ansatz configuration."
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
        help="Getting the number of ansatzes from ansatzes.json",
    )
    args = parser.parse_args()

    # Load json file
    json_file = "ansatzes.json"
    f_ = open(json_file)
    conf = json.load(f_)
    # Creating Combination list
    combination_list = combination_for_list(conf)

    if args.print:
        if args.id is not None:
            print(combination_list[args.id])
        elif args.all:
            print(combination_list)
        else:
            print("Provide -id or --all")
    if args.count:
        print("Number of elements: {}".format(len(combination_list)))
    if args.execution:
        if args.id is not None:
            configuration = combination_list[args.id]
            run_id(**configuration)
        if args.all:
            for configuration in combination_list:
                run_id(**configuration)

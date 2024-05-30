"""
For launching a VQE quantum step execution
Author: Gonzalo Ferro
"""

import sys
import json
sys.path.append("../../")
from PH.utils.utils_ph import combination_for_list
from PH.utils.utils_ph import cartesian_product
from PH.ph_step_exe.vqe_step import run_ph_execution
from PH.qpu.select_qpu import select_qpu


def run_id(**configuration):
    qpu = select_qpu(configuration)
    configuration.update({"qpu":qpu})
    pdf = run_ph_execution(**configuration)
    print(pdf)

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
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()

    parser.add_argument(
        "--count",
        dest="count",
        default=False,
        action="store_true",
        help="Getting the number of elements from vqe_step.json",
    )
    group.add_argument(
        "--all",
        dest="all",
        default=False,
        action="store_true",
        help="Select all the elements from vqe_step.json",
    )
    group.add_argument(
        "-id",
        dest="id",
        type=int,
        help="Select one element from vqe_step.json",
        default=None,
    )
    parser.add_argument(
        "--print",
        dest="print",
        default=False,
        action="store_true",
        help="For printing the configuration."
    )
    parser.add_argument(
        "-json_qpu",
        dest="json_qpu",
        type=str,
        default="../qpu/qpu_ideal.json",
        help="JSON with the qpu configuration",
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

    # Creating Combination list for the VQE steps
    # Load json file
    json_file = "vqe_step.json"
    f_ = open(json_file)
    conf = json.load(f_)
    combination_list = combination_for_list(conf)

    # Creating Combination list for QPUs
    with open(args.json_qpu) as json_file:
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
            run_id(**configuration)
        if args.all:
            for configuration in final_list:
                run_id(**configuration)

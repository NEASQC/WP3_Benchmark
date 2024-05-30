"""
For launching a VQE quantum step execution
Author: Gonzalo Ferro
"""

import sys
import json
sys.path.append("../../")
from PH.utils.utils_ph import combination_for_list
from PH.parent_hamiltonian.parent_hamiltonian import run_parent_hamiltonian


def run_id(**configuration):
    run_parent_hamiltonian(**configuration)

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
        help="Getting the number of elements from parent_hamiltonian.json ",
    )
    group.add_argument(
        "--all",
        dest="all",
        default=False,
        action="store_true",
        help="Select all the elements from parent_hamiltonian.json",
    )
    group.add_argument(
        "-id",
        dest="id",
        type=int,
        help="Select an element from parent_hamiltonian.json",
        default=None,
    )
    parser.add_argument(
        "--print",
        dest="print",
        default=False,
        action="store_true",
        help="For printing the configuration."
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
    # Load json file
    json_file = "parent_hamiltonian.json"
    f_ = open(json_file)
    conf = json.load(f_)
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

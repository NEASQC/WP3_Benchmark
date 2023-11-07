"""
For launching a VQE quantum step execution
Author: Gonzalo Ferro
"""

import json
from utils_ph import combination_for_list
from parent_hamiltonian_mps import run_parent_hamiltonian

def run_id(**configuration):
    run_parent_hamiltonian(**configuration)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        format='%(asctime)s-%(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        #level=logging.INFO
        level=logging.DEBUG
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

    # Load json file
    json_file = "parent_hamiltonian_mps.json"
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

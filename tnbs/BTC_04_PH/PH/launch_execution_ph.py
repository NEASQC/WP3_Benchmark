"""
For launching a VQE quantum step execution
Author: Gonzalo Ferro
"""

import os
import json
from utils import get_filelist
from execution_ph import run_ph_execution

def list_files(folder):
    filelist = os.listdir(folder)
    final_files = []
    for file_ in filelist:
        final_files = final_files + get_filelist(folder + file_+"/")
    return final_files

def run_id(basefn, **configuration):
    configuration.update({"base_fn": basefn})
    pdf = run_ph_execution(**configuration)
    print(pdf)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()

    parser.add_argument(
        "-folder",
        dest="folder",
        type=str,
        default="./",
        help="Path for searching the folder",
    )
    parser.add_argument(
        "--count",
        dest="count",
        default=False,
        action="store_true",
        help="Getting the number of elements",
    )
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
    args = parser.parse_args()
    # Load json file
    json_file = "execution_ph.json"
    f_ = open(json_file)
    conf = json.load(f_)
    print(conf)
    files_list = list_files(args.folder)
    if args.print:
        if args.id is not None:
            print(files_list[args.id])
        elif args.all:
            print(files_list)
        else:
            print("Provide -id or --all")
    if args.count:
        print("Number of elements: {}".format(len(files_list)))
    if args.execution:
        if args.id is not None:
            configuration = files_list[args.id]
            run_id(configuration, **conf)
        if args.all:
            for configuration in files_list:
                run_id(configuration, **conf)

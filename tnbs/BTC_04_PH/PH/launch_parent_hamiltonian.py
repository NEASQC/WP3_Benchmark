"""
For launching a VQE quantum step execution
Author: Gonzalo Ferro
"""

import os
import json
import pandas as pd
from utils import get_filelist
from parent_hamiltonian import run_parent_hamiltonian

def list_files(folder, filelistname):
    #filelist = os.listdir(folder)
    filelist = list(pd.read_csv(filelistname, header=None)[0])
    final_files = []
    for file_ in filelist:
        final_files = final_files + get_filelist(folder + file_+"/")
    return final_files

def run_id(basefn, **configuration):
    state_file = basefn + "_state.csv"
    configuration.update({"base_fn": basefn})
    configuration.update({"state": state_file})
    run_parent_hamiltonian(**configuration)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()

    parser.add_argument(
        "-filelist",
        dest="filelist",
        type=str,
        default="./",
        help="Filename with folder to use",
    )
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
    json_file = "parent_hamiltonian.json"
    f_ = open(json_file)
    conf = json.load(f_)
    print(conf)
    files_list = list_files(args.folder, args.filelist)
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

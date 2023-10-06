"""
For launching a VQE quantum step execution
Author: Gonzalo Ferro
"""

import os
import pandas as pd
from utils_ph import get_filelist
from ansatzes import getting_job

def list_files(filelistname):
    #filelist = os.listdir(folder)
    filelist = pd.read_csv(filelistname, header=None)
    job = filelist[1]
    filename = filelist[0]
    filelist = [get_filelist(file_)[0] for file_ in filename]
    return job, filelist

def run_id(configuration):
    jobid = configuration[0]
    filename = configuration[1]
    conf_dict = {
        "job_id" : jobid,
        "filename": filename,
        "save": True
    }
    state = getting_job(**conf_dict)
    print(state)


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
    parser.add_argument(
        "-filelist",
        dest="filelist",
        type=str,
        default="./",
        help="Filename with folder to use",
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
    #Execution argument
    parser.add_argument(
        "--count",
        dest="count",
        default=False,
        action="store_true",
        help="Getting the number of elements",
    )
    args = parser.parse_args()

    job, files_list = list_files(args.filelist)
    conf_list = list(zip(job, files_list))



    if args.print:
        if args.id is not None:
            print(conf_list[args.id])
        elif args.all:
            print(conf_list)
        else:
            print("Provide -id or --all")
    if args.count:
        print("Number of elements: {}".format(len(conf_list)))
    if args.execution:
        if args.id is not None:
            run_id(conf_list[args.id])
        if args.all:
            for conf_ in conf_list:
                run_id(conf_)

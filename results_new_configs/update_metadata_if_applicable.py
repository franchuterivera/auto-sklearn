import tempfile
from functools import partial
import uuid
import argparse
from random import randrange
import collections
import mmap
import os
import re
import time
import subprocess
import socket
import logging
import glob
import json
import typing
import tqdm

import openml

import sklearn.datasets
from sklearn.utils.multiclass import type_of_target


from shutil import copyfile

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Logger Setup
logger = logging.getLogger('Updater')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manages the run of the benchmark')
    parser.add_argument(
        '--input_json_dir',
        help='From where to take json files',
        required=True
    )
    parser.add_argument(
        '--input_files_dir',
        help='From where to take json files',
        required=True
    )
    parser.add_argument(
        '--output_files_dir',
        help='From where to take json files',
        required=True
    )
    args = parser.parse_args()

    # Get the test score per task
    task2test = {}
    for json_file in tqdm.tqdm(glob.glob(os.path.join(args.input_json_dir, '*.json'))):
        match = re.search(r'(\d+)_(\w+)_1', json_file)
        if match is None:
            raise ValueError(json_file)
        task = match.group(1)
        openml_task = openml.tasks.get_task(task, download_data=False)
        name = openml.datasets.get_dataset(openml_task.dataset_id, download_data=False).name
        metric = match.group(2)
        with open(json_file, 'r') as handle:
            data = json.load(handle)
        if name not in task2test:
            task2test[name] = {}
            task2test[name]['task'] = task
            # Enrich the task structure with openml info
            #X, y = sklearn.datasets.fetch_openml(data_id=openml_task.dataset_id,
            #                                     return_X_y=True, as_frame=False)
            #task2test[name]['target_type'] = type_of_target(y)

        if 'log_loss' in metric:
            task2test[name][metric] = data['test']
        else:
            task2test[name][metric] = 1 - data['test']

    with open('data.json', 'w') as fp:
        json.dump(task2test, fp, indent=4, sort_keys=True)
    #with open('data.json', 'r') as fp:
    #    task2test = json.load(fp)

    new_id = max([pd.read_csv(filename)['idx'].max() for filename in glob.glob(
        os.path.join(args.input_files_dir, '*', 'configurations.csv'))]) + 1

    # We add this new configuration as a new index
    #new_configuration = pd.read_csv('new_configurations.csv', index_col='idx')
    #new_configuration.index = [new_id]
    new_configuration = 'INDEX,weighting,mlp,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,relu,0.0001,auto,0.9,0.999,train,1E-08,2,0.0003,32,128,True,adam,0.0001,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,one_hot_encoding,minority_coalescer,0.004071801722749603,median,quantile_transformer,1000,normal,,,no_preprocessing,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'.replace('INDEX', str(new_id))

    # Update the algorithms/config files when applicable
    for filename in sorted(glob.glob(os.path.join(args.input_files_dir, '*'))):
        if 'sparse' in filename:
            # Only care about dense datasets
            continue
        if 'regression' in filename:
            # Only care about classification datasets
            continue
        change = False

        # "accuracy_binary.classification_dense"
        try:
            metric_target_type, task_type_sparse = os.path.basename(filename).split('.')
        except Exception as e:
            print(filename)
            raise e
        if 'binary' in filename:
            metric = metric_target_type.replace('_binary', '')
            task_type = 'binary'
        elif 'multiclass' in filename:
            metric = metric_target_type.replace('_multiclass', '')
            task_type = 'multiclass'
        else:
            # No regression
            continue

        # @DATA
        # twonorm,1.0,1,0.015561015561015523,ok
        with open(os.path.join(filename, 'algorithm_runs.arff')) as handle:
            contents = [line.strip() for line in handle.readlines()]
        data_done = False
        try:
            with open(os.path.join(filename, 'algorithm_runs.arff').replace(args.input_files_dir, args.output_files_dir), 'w') as handle:
                for i, line in enumerate(contents):
                    if '@DATA' in line:
                        data_done = True
                    elif data_done:
                        instance_id, repetition, algorithm, loss, runstatus = line.split(',')
                        if instance_id in task2test and metric in task2test[instance_id] and 'ok' in runstatus:
                            new_loss = task2test[instance_id][metric]
                            if float(new_loss) < float(loss):
                                line = f"{instance_id},{repetition},{new_id},{new_loss},{runstatus}"
                                print(f"Found better loss={float(new_loss) < float(loss)} for {instance_id} on {filename}")
                                change = True
                    handle.write(f"{line}\n")
        except Exception as e:
            print(os.path.join(filename, 'algorithm_runs.arff'), line)
            raise e

        if change:
            old_configuration_path = os.path.join(filename, 'configurations.csv')
            with open(old_configuration_path, 'r') as handle:
                contents = [line.strip() for line in handle.readlines()]

            with open(old_configuration_path.replace(args.input_files_dir, args.output_files_dir), 'w') as handle:
                for line in contents:
                    handle.write(f"{line}\n")
                handle.write(f"{new_configuration}\n")

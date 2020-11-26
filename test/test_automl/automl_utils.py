# -*- encoding: utf-8 -*-
import re
import os
import glob

import numpy as np


def print_debug_information(automl):

    # In case it is called with estimator,
    # Get the automl object
    if hasattr(automl, 'automl_'):
        automl = automl.automl_

    # Log file path
    log_file = glob.glob(os.path.join(
        automl._backend.temporary_directory, 'AutoML*.log'))[0]

    include_messages = ['INFO', 'DEBUG', 'WARN',
                        'CRITICAL', 'ERROR', 'FATAL']

    # There is a lot of content in the log files. Only
    # parsing the main message and ignore the metalearning
    # messages
    try:
        with open(log_file) as logfile:
            content = logfile.readlines()

        # Get the messages to debug easier!
        content = [x for x in content if any(
            msg in x for msg in include_messages
        ) and 'metalearning' not in x]

    except Exception as e:
        return str(e)

    # Also add the run history if any
    if hasattr(automl, 'runhistory_') and hasattr(automl.runhistory_, 'data'):
        for k, v in automl.runhistory_.data.items():
            content += ["{}->{}".format(k, v)]
    else:
        content += ['No RunHistory']

    # Also add the ensemble history if any
    if len(automl.ensemble_performance_history) > 0:
        content += [str(h) for h in automl.ensemble_performance_history]
    else:
        content += ['No Ensemble History']

    return os.linesep.join(content)


def count_succeses(cv_results):
    return np.sum(
        [status in ['Success', 'Success (but do not advance to higher budget)']
         for status in cv_results['status']]
    )


class AutoMLLogParser(object):
    def __init__(self, logfile):
        self.logfile = logfile
        self.lines = self.parse_logfile()

    def parse_logfile(self):
        # We care about the [debug/info/...] messages
        assert os.path.exists(self.logfile), "{} not found".format(self.logfile)

        with open(self.logfile) as f:
            content = f.readlines()
        content = [x.strip() for x in content if re.search(r'[\w+]', x)]
        return content

    def count_ensembler_iterations(self):
        # we have to count the start msg from pynisher
        # and the return msg
        # We expect the start msg to be something like:
        # [DEBUG] [2020-11-26 19:22:42,160:EnsembleBuilder] Function called with argument: (61.
        iterations = []

        # One thing is to call phynisher, the other is to actually execute the funciton
        iterations_from_inside_ensemble_builder = []
        for line in self.lines:

            # Pynisher call
            match = re.search(
                r'EnsembleBuilder]\s+Function called with argument:\s+\(\d+\.\d+, (\d+), \w+',
                line)
            if match:
                iterations.append(int(match.group(1)))

            # Ensemble Builder actuall call
            match = re.search(
                r'EnsembleBuilder]\s+Starting iteration (\d+)',
                line)
            if match:
                iterations_from_inside_ensemble_builder.append(int(match.group(1)))

        assert iterations == iterations_from_inside_ensemble_builder, "{} ! {}".format(
            iterations, iterations_from_inside_ensemble_builder
        )

        return iterations

    def count_ensembler_success_pynisher_calls(self):

        # We expect the return msg to be something like:
        # [DEBUG] [2020-11-26 19:22:43,018:EnsembleBuilder] return value: (([{'Times...
        return_msgs = len([x for x in self.lines if re.search(
            r'EnsembleBuilder]\s+return value:.*Timestamp', x)])

        return return_msgs

    def count_tae_pynisher_calls(self):
        # We expect the return msg to be something like:
        # [DEBUG] [2020-11-26 19:22:39,558:pynisher] return value: (...
        return_msgs = len([x for x in self.lines if re.search(
            r'pynisher]\s+return value:\s+', x)])
        return (return_msgs)

    def get_automl_setting_from_log(self, dataset_name, setting):
        for line in self.lines:
            match = re.search(
                f"{dataset_name}]\\s*{setting}\\s*:\\s*(\\w+)",
                line)
            if match:
                return match.group(1)
        return None

import argparse
import json
import logging
import time
import os
import numpy as np

from routines import TherapiesRecommender

# Parsing the command-line arguments
parser = argparse.ArgumentParser(description='Recommends therapies for a certain condition of a patient in a dataset.')
parser.add_argument('datasetPath', help='Position of the json file with data.')
parser.add_argument('--cases', dest='casesPath', help='Position of the txt file containing multiple cases.')
parser.add_argument('--patient', dest='patientId', help='ID of the patient.')
parser.add_argument('--patient-cond', dest='patientCondId', help='ID of the uncured condition of the patient.')
parser.add_argument('--evaluate', '-e', action='store_true', help='Perform A/B eval on the data')
parser.add_argument('-v', action='store_true', help='Display more execution information')

args = parser.parse_args()

if args.v:
    logging.basicConfig(level=logging.INFO)

# == Loading the dataset ==
logging.info('Loading the dataset...')
s_time = time.time()
with open(args.datasetPath, 'r') as file:
    str_dataset = file.read().replace('\n', '')
recommender = TherapiesRecommender(json.loads(str_dataset))
logging.info(f"Loading the dataset: DONE! [{time.time() - s_time:.2f}s]")

if args.evaluate:
    logging.info(f"Evaluation...")
    s_time = time.time()
    iterations = 100
    experiment = recommender.evaluate(folds=1000, iterations=iterations, lsh_rows=32, lsh_bands=400, svd_thresh=.9,
                                      untried_success=5, pa_k=3, cond_k=3)
    logging.info(f"Evaluation: DONE! [{(time.time() - s_time) / iterations:.2f}s/it]")
    mode = 'a' if os.path.exists('../bin/performances.txt') else 'w+'
    with open('../bin/performances.txt', mode) as file:
        if mode == 'w+':
            header = '\t'.join(experiment.keys()) + '\n'
            file.write(header)
        exp_vals = '\t\t'.join(
            np.format_float_positional(val, precision=2, trim='-') for val in experiment.values()) + '\n'
        file.write(exp_vals)
else:
    # == Building the ratings tensor ==
    recommender.build_ratings_tensor()
    # == Global baseline ==
    recommender.build_global_baseline()
    # == Building the nearest neighbors ==
    recommender.build_nearest_neighbors()
    # == Predicting the cases ==
    if args.casesPath is not None:
        with open(args.casesPath, 'r') as file:
            cases = [tuple(line.split()) for line in file.readlines()][1:]
    elif args.patientId is not None and args.patientCondId is not None:
        cases = [(args.patientId, args.patientCondId)]
    else:
        cases = []

    if len(cases) > 0:
        solutions = recommender.predict(cases)
        f_name = '.'.join(os.path.basename(args.casesPath).split('.')[:-1])
        with open(f"../results/{f_name}_sol.txt", 'w+') as file:
            for sol in solutions:
                file.write(f"{' '.join(sol)}\n")

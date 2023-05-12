from itertools import groupby
import logging
import time
import numpy.ma as ma

from algorithms import *


class TherapiesRecommender:
    def __init__(self, dataset):
        self.data_mask = None
        self.curr_discount = None
        self.curr_untried_success = None
        self.g_baseline = None
        self._eval = False
        self.nearestCondSim = None
        self.nearestCond = None
        self.nearestPaSim = None
        self.nearestPa = None
        self.ratings = None
        self.paCond2thIdx = defaultdict(list)
        self.dataset = dataset
        self.cond2idx = {condition['id']: idx for (idx, condition) in enumerate(dataset['Conditions'])}
        self.pa2idx = {str(patient['id']): idx for (idx, patient) in
                       enumerate(filter(suitable_patient, dataset['Patients']))}
        self.th2idx = {therapy['id']: idx for (idx, therapy) in enumerate(dataset['Therapies'])}
        self.idx2th = list(self.th2idx.keys())
        self.paCond2Cond = {paCond['id']: paCond['kind'] for pa in dataset['Patients'] for paCond in pa["conditions"]}
        self.conditions = len(self.cond2idx)
        self.patients = len(self.pa2idx)
        self.therapies = len(self.th2idx)
        # Sort the patient trials according to: condition, patient condition, start date
        for patient in self.dataset['Patients']:
            patient['trials'] = [trial for trial in patient['trials'] if isinstance(trial['successful'], (int, float))]
            patient['trials'].sort(key=lambda tr: (self.paCond2Cond[tr['condition']], tr['condition'], tr['start']))

    def evaluate(self, folds=20, iterations=None, k=5, untried_success=5, discount=.9, svd_thresh=.9, lsh_bands=300,
                 lsh_rows=26, pa_k=10, cond_k=5):
        self._eval = True
        iterations = folds if iterations is None or iterations > folds else iterations
        fold_assign = np.random.randint(0, folds, (self.conditions, self.patients), dtype='uint16')
        self.build_ratings_tensor(untried_success=untried_success, discount=discount)
        experiment = {
            f"p@{k} %": [],
            "NN build [s]": []
        }

        for i, fold in enumerate(range(iterations)):
            logging.info(f"Evaluating fold [{i + 1}/{iterations}]")
            # Which (Cond, Pa) elements are test elements
            test_mask = fold_assign == fold
            # Reset the mask
            self.ratings.mask = ma.nomask
            self._build_data_mask()
            # Prepare the cases
            cases = [(pa_idx, cond_idx) for cond_idx, pa_idx in zip(*np.where(test_mask & self.data_mask))]
            logging.info(f"Number of cases: {len(cases)}")
            # Ground truth
            ground_truths = [best_therapies(self.ratings[cond_idx, pa_idx], keep=k) for pa_idx, cond_idx in cases]
            # Mask the ratings for the prediction
            self.ratings.mask = np.broadcast_to(test_mask[..., np.newaxis], self.ratings.shape)
            self._build_data_mask()
            # Build the engine
            self.build_global_baseline()
            nn_time = time.time()
            self.build_nearest_neighbors(svd_thresh=svd_thresh, lsh_bands=lsh_bands, lsh_rows=lsh_rows, pa_k=pa_k,
                                         cond_k=cond_k)
            experiment["NN build [s]"].append(time.time() - nn_time)
            # Predict the cases
            predictions = self.predict(cases, k)
            # Evaluate
            fold_precision = np.array([precision_at_k(pred, gt) for pred, gt in zip(predictions, ground_truths)]).mean()
            experiment[f"p@{k} %"].append(fold_precision)
        self.ratings.mask = ma.nomask
        self._build_data_mask()
        self._eval = False
        # Take the average values
        experiment = {metric: np.array(scores).mean() for metric, scores in experiment.items()}
        # Give the experiment values
        experiment[f"p@{k} %"] *= 100
        experiment["# of Folds"] = folds
        experiment["# Iterations"] = iterations
        experiment["UntriedSuccess"] = untried_success
        experiment["Discount"] = discount
        experiment["SVD Thresh"] = svd_thresh
        experiment["# LSH Bands"] = lsh_bands
        experiment["# LSH Rows"] = lsh_rows
        experiment["# Patient NN"] = pa_k
        experiment["# Cond NN"] = cond_k

        return experiment

    def build_ratings_tensor(self, untried_success=5, discount=.9, del_dataset=True):
        if self.curr_untried_success == untried_success and self.curr_discount == discount and self.ratings is not None:
            logging.info("Reusing Ratings tensor. No need to rebuild.")
            self.ratings.mask = ma.nomask
            self._build_data_mask()
            return
        logging.info('Building the ratings tensor...')
        s_time = time.time()

        self.curr_untried_success = untried_success
        self.curr_discount = discount
        self.ratings = ma.zeros((self.conditions, self.patients, self.therapies), fill_value=0, dtype='int8')
        for patient in filter(suitable_patient, self.dataset['Patients']):
            # Group together trials for the same medical condition
            for condId, condTrials in groupby(patient['trials'], key=lambda i: self.paCond2Cond[i['condition']]):
                therapies_ratings = np.full(self.therapies, fill_value=untried_success)
                therapies_counts = np.zeros(self.therapies, dtype='int')
                # Group together the trials for the same patient condition (individual instance of the condition)
                for paCond, paCondTrials in groupby(condTrials, key=lambda i: i['condition']):
                    # Convert the trials for the patient condition into a {'therapy': rating, ...} dict
                    pa_cond_therapies = trials2therapies(paCondTrials, discount=discount)
                    # Add the paCond therapies to the condition row
                    for therapy, score in pa_cond_therapies.items():
                        th_idx = self.th2idx[therapy]
                        # Keep a registry of all past therapies indexes
                        self.paCond2thIdx[paCond].append(th_idx)
                        # Do not consider untried values in the rating average, only tried therapies
                        prev_rating = 0 if therapies_counts[th_idx] == 0 else therapies_ratings[th_idx]
                        therapies_ratings[th_idx] = prev_rating + score
                        therapies_counts[th_idx] += 1
                # Average the therapies_ratings where the therapies_counts are > 1
                mask = therapies_counts > 1
                therapies_ratings[mask] = therapies_ratings[mask] / therapies_counts[mask]
                # Normalization: extend the range to 127 and subtract the average score for untried therapies
                therapies_ratings = ((therapies_ratings - untried_success) * (127 / 100)).astype('int8')
                # Add the therapies scores for the condition of the patient to the ratings
                cond_idx = self.cond2idx[condId]
                pa_idx = self.pa2idx[str(patient['id'])]
                self.ratings[cond_idx, pa_idx] += therapies_ratings

        self._build_data_mask()

        if del_dataset:
            del self.dataset

        ratings_gb = self.ratings.nbytes / (1000 ** 3)
        logging.info(f"Building the ratings tensor: DONE! ({ratings_gb:.2f}GB) [{time.time() - s_time:.2f}s]")

    def _build_data_mask(self):
        self.data_mask = (self.ratings != 0).any(axis=2).filled(False)  # (Cond, Pa): If there is data

    def build_global_baseline(self):
        logging.info('Computing the global baseline...')
        s_time = time.time()

        non_zero_mask = self.ratings != 0

        pa_sum = self.ratings.sum((0, 2))
        non_zero_pa = non_zero_mask.sum((0, 2))
        non_zero_pa[non_zero_pa == 0] = 1  # Divide by 1 where all values are 0
        pa_avg = (pa_sum / non_zero_pa).filled().astype('float16')

        th_sum = self.ratings.sum((0, 1))
        non_zero_th = non_zero_mask.sum((0, 1))
        non_zero_th[non_zero_th == 0] = 1
        th_avg = (th_sum / non_zero_th).filled().astype('float16')

        avg = (th_sum.sum() / non_zero_th.sum()).astype('float16')  # Exploit computations from the th_avg

        pa_deviation = pa_avg - avg
        th_deviation = th_avg - avg
        g_baseline = (pa_deviation[..., np.newaxis] + th_deviation[np.newaxis, ...]) + avg  # Shape: (Pa, Th)
        self.g_baseline = g_baseline[np.newaxis, :, :]  # Shape: (1, Pa, Th)

        baseline_gb = self.g_baseline.nbytes / (1000 ** 3)
        logging.info(f"Computing the global baseline: DONE! ({baseline_gb:.2f}GB) [{time.time() - s_time:.2f}s]")

    def build_nearest_neighbors(self, svd_thresh=.9, lsh_bands=300, lsh_rows=26, pa_k=10, cond_k=5):
        logging.info('Computing patients and cond. nearest neighbors...')
        s_time = time.time()

        # Nearest patients: two patients are similar if are given similar therapies and react similarly
        matrix = self.ratings.filled().mean(0).astype('float32')  # (Pa, Th)
        good_mask = self.data_mask.any(0)  # (Cond, Pa) -> (Pa)
        self.nearestPa, self.nearestPaSim = self._nn(matrix, svd_thresh, lsh_bands, lsh_rows, pa_k, good_mask)
        # Conditions embeddings: two conditions are similar if they are treated with similar therapies
        matrix = self.ratings.filled().mean(1).astype('float32')  # (Cond, Th)
        good_mask = self.data_mask.any(1)  # (Cond, Pa) -> (Cond)
        self.nearestCond, self.nearestCondSim = self._nn(matrix, svd_thresh, lsh_bands, lsh_rows, cond_k, good_mask)

        nn_gb = (self.nearestPa.nbytes + self.nearestPaSim.nbytes + self.nearestCond.nbytes + self.nearestCondSim.nbytes) / (1000 ** 3)
        logging.info(f"Computing patients and cond. nearest neighbors: DONE! ({nn_gb:.2f}GB) [{time.time() - s_time:.2f}s]")

    @staticmethod
    def _nn(matrix, svd_thresh, lsh_bands, lsh_rows, k, good_mask):
        all_entities_count = matrix.shape[0]
        good_entities = np.argwhere(good_mask).squeeze()
        embeddings = svd_embeddings(matrix[good_entities, :], threshold=svd_thresh, mode='linear')
        candidate_pairs = lsh_cosine(embeddings, lsh_bands, lsh_rows)
        return top_k_similar(embeddings, candidate_pairs, k, good_entities, all_entities_count)

    def predict(self, cases, k=5, ignore_tried_therapies=False):
        logging.info('Predicting the cases...')
        s_time = time.time()

        cases_solutions = []

        ratings = self.ratings.filled()
        pa_mask = self.data_mask.any(0)
        cond_mask = self.data_mask.any(1)

        for paId, paCondId in cases:
            if self._eval:
                # Interpret directly the passed cases as indices
                pa_idx = paId
                cond_idx = paCondId
                tried_therapies = None
            else:
                # If the patient has no ratings, average all the patients equally
                pa_idx = self.pa2idx[paId] if paId in self.pa2idx else None
                cond_id = self.paCond2Cond[paCondId] if paCondId in self.paCond2Cond else None
                cond_idx = self.cond2idx[cond_id] if cond_id in self.cond2idx else None
                tried_therapies = None if ignore_tried_therapies else self.paCond2thIdx[paCondId]
            # Check if the indices have been removed because not good entities (It is better to average)
            pa_idx = pa_idx if pa_idx is not None and pa_mask[pa_idx] else None  # (Pa)
            cond_idx = cond_idx if cond_idx is not None and cond_mask[cond_idx] else None  # (Cond)

            pred_therapies = predict_therapies(ratings, self.g_baseline, self.nearestPa, self.nearestPaSim,
                                               self.nearestCond, self.nearestCondSim, cond_idx, pa_idx)
            top_therapies = best_therapies(pred_therapies, keep=k, tried_therapies=tried_therapies)
            if not self._eval:
                # Convert the therapies indices into their ID
                top_therapies = [self.idx2th[th] for th in top_therapies]
            cases_solutions.append(top_therapies)

        logging.info(f"Predicting the cases: DONE! [{time.time() - s_time:.2f}s]")

        return cases_solutions

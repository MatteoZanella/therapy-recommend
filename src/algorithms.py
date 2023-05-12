import itertools
import heapq
from collections import Counter, defaultdict
import numpy as np


def suitable_patient(patient):
    return len(patient['trials']) > 0


def trials2therapies(trials, mode='discount', discount=.9):
    """
    Transforms an ordered iterable of trials to a dictionary {therapy: rating, ...}.
    :param trials: Sorted list of trials
    :param mode: ('discount') Controls how the rating is computed
    :param discount: Discount factor in [0,1] for the discount mode
    :return: Dictionary where keys are therapies and values are their rating
    """
    occurrences = Counter()
    ratings_sum = Counter()
    for order, trial in enumerate(trials):
        successful = trial['successful']
        if mode == 'discount':
            score = successful * (discount ** order)
        else:
            score = successful
        occurrences.update([trial['therapy']])
        ratings_sum.update({trial['therapy']: score})
    # Average the scores if the same therapy has been tried multiple times for the same patient condition
    return {therapy: ratings_sum[therapy] / occurrences[therapy] for therapy in ratings_sum}


def cosine_similarity(u, v):
    norms = np.linalg.norm(u) * np.linalg.norm(v)
    if norms == 0:
        return norms
    else:
        return (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))


def svd_embeddings(ratings, threshold, mode='squared'):
    """
    Computes the embeddings for the elements along the first axis with SVD decomposition
    :param ratings: The ratings matrix
    :param threshold: Maximum percentage of energy to maintain in the approximation, ranging in [0,1]
    :param mode: ('squared'|'linear') How to modify the values to get their energy
    :return: The embeddings matrix
    """
    embeddings, sigma, _ = np.linalg.svd(ratings, full_matrices=False)
    latent_size = best_approximation(sigma, threshold, mode=mode)
    return embeddings[:, :latent_size]


def best_approximation(eigenvalues, threshold, mode='squared'):
    """
    Returns the number of singular values holding the most significance
    :param eigenvalues: NumPy array of eigenvalues
    :param threshold: [0,1] Minimum percentage of energy to maintain
    :param mode: ('squared'|'linear') How to modify the values to get their energy
    :return: The number of significant eigenvalues
    """
    if mode == 'squared':
        eigenvalues = eigenvalues ** 2
    cum_sum = np.cumsum(eigenvalues)
    return np.argmax(cum_sum >= cum_sum[-1] * threshold) + 1


def lsh_cosine(embeddings, b, r):
    """
    LSH technique for cosine similarity
    :param embeddings: (n, d) matrix with the embeddings in the rows
    :param b: Number of bands
    :param r: Rows per band, maximum of 32
    :return: The list of candidate pairs above the chosen threshold
    """
    f = embeddings.shape[1]
    hyperplanes = np.random.randn(f, b * r)
    signatures = (embeddings @ hyperplanes) > 0
    # Divide into bands
    banded_signatures = np.split(signatures, b, axis=1)
    # Transform each band into an integer
    binary_column = 2 ** np.arange(r, dtype='uint32' if r > 16 else 'uint16')[:, np.newaxis]
    signatures = np.hstack([band @ binary_column for band in banded_signatures])
    # Put into hash buckets the bands values
    hash_buckets = defaultdict(set)
    with np.nditer(signatures, flags=['multi_index']) as it:
        for x in it:
            emb_idx, band_idx = it.multi_index
            hash_buckets[(band_idx, x.item())].add(emb_idx)
    # Find the candidate pairs
    candidate_pairs = set()
    for p, bucket in hash_buckets.items():
        if len(bucket) > 1:
            for pair in itertools.combinations(bucket, 2):
                candidate_pairs.add(pair)
    return candidate_pairs


def top_k_similar(embeddings, candidate_pairs, k, good_entities, all_entities_count, similarity='cosine'):
    neighbors = [[] for _ in range(all_entities_count)]
    for u, v in candidate_pairs:
        similarity = cosine_similarity(embeddings[u], embeddings[v])
        real_u = good_entities[u]
        real_v = good_entities[v]
        heapq.heappush(neighbors[real_u], (similarity, real_v))
        heapq.heappush(neighbors[real_v], (similarity, real_u))
    neighbors = [heapq.nlargest(k, heap) for heap in neighbors]
    real_k = max([len(heap) for heap in neighbors])
    # If there aren't enough neighbors, add fake ones with sim=0 and u=0
    neighbors = [[(sim, v) for (sim, v), _ in itertools.zip_longest(heap, range(real_k), fillvalue=(0, 0))] for heap in
                 neighbors]
    top_similarities = np.array([[sim for (sim, v) in heap] for heap in neighbors], dtype='float16')
    top_neighbors = np.array([[v for (sim, v) in heap] for heap in neighbors], dtype='uint')
    top_similarities = np.hstack([np.ones((top_similarities.shape[0], 1), dtype='float16'), top_similarities])
    top_neighbors = np.hstack([np.arange(top_neighbors.shape[0], dtype='uint')[..., np.newaxis], top_neighbors])
    return top_neighbors, top_similarities


def predict_therapies(ratings, g_baseline, near_pa, near_pa_sim, near_cond, near_cond_sim, cond_idx, pa_idx):
    nearest_patients = ... if pa_idx is None else near_pa[pa_idx]
    nearest_conditions = ... if cond_idx is None else near_cond[cond_idx]
    near_g_baseline = g_baseline[:, nearest_patients]  # (1, NearPa/Pa, Th)
    near_ratings = ratings[:, nearest_patients][nearest_conditions]  # (NearCond/Cond, NearPa/Pa, Th)

    if pa_idx is not None and cond_idx is not None:
        weights = np.outer(near_cond_sim[cond_idx], near_pa_sim[pa_idx])[..., np.newaxis]  # (NearCond, NearPa, 1)
        avg_th = ((near_ratings - near_g_baseline) * weights).sum(axis=(0, 1)) / weights.sum()  # (Th)
        patient_baseline = g_baseline[:, pa_idx].squeeze()  # (1, 1, Th) -> (Th)
    elif pa_idx is None and cond_idx is None:
        avg_g_baseline = near_g_baseline.mean(axis=(0, 1))  # (Th)
        near_ratings = near_ratings  # (Cond, Pa, Th)
        avg_th = near_ratings.mean(axis=(0, 1)) - avg_g_baseline  # (Th)
        patient_baseline = avg_g_baseline.squeeze()  # (1, Th) -> (Th)
    elif pa_idx is None:
        avg_g_baseline = near_g_baseline.mean(axis=1)  # (1, Th)
        near_ratings = near_ratings.mean(axis=1)  # (NearCond, Th)
        weights = near_cond_sim[cond_idx][..., np.newaxis]  # (NearCond, 1)
        avg_th = ((near_ratings - avg_g_baseline) * weights).sum(axis=0) / weights.sum()  # (Th)
        patient_baseline = avg_g_baseline.squeeze()  # (1, Th) -> Th
    else:
        avg_g_baseline = near_g_baseline.mean(axis=1)  # (1, Th)
        near_ratings = near_ratings.mean(axis=0)  # (NearPa, Th)
        weights = near_pa_sim[pa_idx][..., np.newaxis]  # (NearPa, 1)
        avg_th = ((near_ratings - avg_g_baseline) * weights).sum(axis=0) / weights.sum()  # (Th)
        patient_baseline = g_baseline[:, pa_idx].squeeze()  # (1, 1, Th) -> (Th)

    return avg_th + patient_baseline


def best_therapies(candidate_therapies, keep=5, tried_therapies=None):
    """ Takes a vector of predicted therapies and gives the best ones"""
    if tried_therapies is None:
        tried_therapies = []
    # Take therapies indexes from best to worst
    top_therapies = np.argsort(-candidate_therapies)[:keep + len(tried_therapies)]
    # Filter out already tried therapies
    if len(tried_therapies) > 0:
        top_therapies = top_therapies[~np.isin(top_therapies, tried_therapies)][:keep]
    return top_therapies


def precision_at_k(prediction, ground_truth):
    return np.isin(prediction, ground_truth).sum() / len(prediction)

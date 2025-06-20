import copy
import difflib
import json
import time
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def cluster_strings(strings: list[str], threshold: float = 0.9) -> list[list[str]]:
    clusters = []
    cluster_num = 0
    while len(strings) > 0:
        string = strings.pop()
        clusters.append([string])
        indexes_to_remove = []
        for idx, string_to_match in enumerate(strings):
            if difflib.SequenceMatcher(None, string, string_to_match).quick_ratio() > threshold:
                clusters[cluster_num].append(string_to_match)
                indexes_to_remove.append(idx)
        strings = [string for idx, string in enumerate(strings) if idx not in indexes_to_remove]
        cluster_num += 1
    return clusters


def cluster_strings_cached(strings: list[str], threshold: float = 0.9) -> list[list[str]]:
    clusters = []
    cluster_num = 0
    cache: dict[tuple[str, str], float] = {}
    while len(strings) > 0:
        string = strings.pop()
        clusters.append([string])
        indexes_to_remove = []
        for idx, string_to_match in enumerate(strings):
            if (string, string_to_match) not in cache:
                ratio = difflib.SequenceMatcher(None, string, string_to_match).quick_ratio()
                cache[(string, string_to_match)] = ratio
            else:
                ratio = cache[(string, string_to_match)]
            if ratio > threshold:
                clusters[cluster_num].append(string_to_match)
                indexes_to_remove.append(idx)
        strings = [string for idx, string in enumerate(strings) if idx not in indexes_to_remove]
        cluster_num += 1
    cache.clear()
    return clusters


def _cluster_strings_exhaustive_internal(
    strings: list[str], threshold: float, similarity_matrix: npt.NDArray[np.float64]
) -> list[list[str]]:
    # Remove negatives values.
    # We don't care about vector direction, only magnitude (how different are the strings).
    # Similarity vectors from string data rarely contain negative values anyways.
    similarity_matrix = np.abs(similarity_matrix)
    # Calculate distance matirx manually instead of calling cosine_distances because of the previous step.
    # Also remove negatives, just in case, there are cases where we get very small negative values.
    distance_matrix = np.abs(1 - similarity_matrix)
    # Calculate clusters.
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, metric="precomputed", linkage="average"
    )
    labels = clustering.fit_predict(distance_matrix)
    # Group strings into clusters.
    clusters: dict[object, list[str]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(strings[idx])
    return list(clusters.values())


def cluster_strings_exhaustive(strings: list[str], threshold: float = 0.75) -> list[list[str]]:
    # Vectorize srings.
    vectorizer = TfidfVectorizer()
    document_term_matrix = vectorizer.fit_transform(strings)
    # Compute similarity matrix.
    similarity_matrix = cosine_similarity(document_term_matrix)
    # Compute clusters and groups strings into clusters.
    return _cluster_strings_exhaustive_internal(strings, threshold, similarity_matrix)


def _get_similarity_matrix_diff_ratios(strings: list[str]) -> npt.NDArray[np.float64]:
    n = len(strings)
    similarity_matrix = np.empty((n, n))
    cache: dict[tuple[str, str], float] = {}
    # The matrix ends up being symetrical because quick_ratio returns
    # the same value independently of the order of the inputs.
    # So we only calculate on half the values and mirror them into the other half.
    for i in range(n):
        similarity_matrix[i][i] = 1.0  # Diagonal is always 1.0 (strings are equal).
        for j in range(i + 1, n):
            if (strings[i], strings[j]) not in cache:
                ratio = difflib.SequenceMatcher(None, strings[i], strings[j]).quick_ratio()
                cache[(strings[i], strings[j])] = ratio
            else:
                ratio = cache[(strings[i], strings[j])]
            similarity_matrix[i][j] = ratio
            similarity_matrix[j][i] = ratio  # Mirror the value
    cache.clear()
    return similarity_matrix


def cluster_strings_exhaustive_ratio(strings: list[str], threshold: float = 0.1) -> list[list[str]]:
    # Compute similarity matrix.
    similarity_matrix = _get_similarity_matrix_diff_ratios(strings)
    # Compute clusters and groups strings into clusters.
    return _cluster_strings_exhaustive_internal(strings, threshold, similarity_matrix)


def _globify_strings_internal(strings: list[str]) -> str:
    result = strings[0]
    for string in strings[1:]:
        # Reverse the opcodes so that indexes don't change as the operations are applied.
        for tag, i1, i2, _j1, _j2 in reversed(difflib.SequenceMatcher(None, result, string).get_opcodes()):
            if tag != "equal":
                result = result[:i1] + "*" + result[i2:]
    return result


def globify_strings(strings: list[str]) -> str:
    if len(strings) == 0:
        return ""
    strings = list(set(strings))  # Remove duplicates.
    if len(strings) == 1:
        return strings[0]
    return _globify_strings_internal(strings)


def _globify_strings_internal_cached(strings: list[str], cache: dict[tuple[str, str], str]) -> str:
    result = strings[0]
    for string in strings[1:]:
        if (result, string) not in cache:
            result_prev = copy.copy(result)
            # Reverse the opcodes so that indexes don't change as the operations are applied.
            for tag, i1, i2, _j1, _j2 in reversed(difflib.SequenceMatcher(None, result, string).get_opcodes()):
                if tag != "equal":
                    result = result[:i1] + "*" + result[i2:]
            cache[(result_prev, string)] = result
        else:
            result = cache[(result, string)]
    return result


def globify_strings_exhaustive(strings: list[str]) -> str:
    if len(strings) == 0:
        return ""
    strings = list(set(strings))  # Remove duplicates.
    if len(strings) == 1:
        return strings[0]
    results = []
    cache: dict[tuple[str, str], Any] = {}
    for i, string in enumerate(strings):
        cmp_strings = [string] + strings[:i] + strings[i + 1 :]
        results.append(_globify_strings_internal_cached(cmp_strings, cache))
    return max(results, key=len)


if __name__ == "__main__":
    from test_data import strings1

    strings = strings1

    start_t = time.time()
    clustered_strings = cluster_strings(strings)
    print("cluster_strings", time.time() - start_t)

    # print(json.dumps(clustered_strings, indent=2))

    start_t = time.time()
    clustered_strings = cluster_strings_cached(strings)
    print("cluster_strings_cached", time.time() - start_t)

    # print(json.dumps(clustered_strings, indent=2))

    start_t = time.time()
    clustered_strings = cluster_strings_exhaustive(strings)
    print("cluster_strings_exhaustive", time.time() - start_t)

    # print(json.dumps(clustered_strings, indent=2))

    start_t = time.time()
    clustered_strings = cluster_strings_exhaustive_ratio(strings)
    print("cluster_strings_exhaustive_ratio", time.time() - start_t)

    # print(json.dumps(clustered_strings, indent=2))

    # for cluster in clustered_strings:
    #     for i, string in enumerate(cluster):
    #         cluster[i] = glob.escape(string)

    start_t = time.time()
    globbed_strings = []
    for cluster in clustered_strings:
        globbed_strings.append(globify_strings(cluster))
    print("globify_strings", time.time() - start_t)

    print(json.dumps(globbed_strings, indent=2))

    start_t = time.time()
    globbed_strings = []
    for cluster in clustered_strings:
        globbed_strings.append(globify_strings_exhaustive(cluster))
    print("globify_strings_exhaustive", time.time() - start_t)

    print(json.dumps(globbed_strings, indent=2))

    # print(globify_strings(["ab", "ba", "cb", "bc"]))
    # print(globify_strings_exhaustive(["ab", "ba", "cb", "bc"]))
    # print(
    #     globify_strings(
    #         [
    #             "ERROR: Some extra message. Bla bla bla bla.",
    #             "ERROR: Bla bla bla bla. Some extra message.",
    #             "ERROR: Bla bla bla bla.",
    #         ]
    #     )
    # )
    # print(
    #     globify_strings_exhaustive(
    #         [
    #             "ERROR: Some extra message. Bla bla bla bla.",
    #             "ERROR: Bla bla bla bla. Some extra message.",
    #             "ERROR: Bla bla bla bla.",
    #         ]
    #     )
    # )

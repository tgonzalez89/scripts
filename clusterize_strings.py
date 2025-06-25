import copy
import datetime
import difflib
import json
import os
import random
import re
import time
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path
from string import ascii_letters, digits
from typing import Any

import Levenshtein
import numpy as np
import numpy.typing as npt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class StrCmp(Enum):
    DIFFLIB = auto()
    LEVENSHTEIN = auto()


_STR_CMP = StrCmp.LEVENSHTEIN


################################################################################


def _get_ratio(str1: str, str2: str) -> float:
    match _STR_CMP:
        case StrCmp.DIFFLIB:
            return difflib.SequenceMatcher(None, str1, str2).quick_ratio()
        case StrCmp.LEVENSHTEIN:
            return Levenshtein.ratio(str1, str2)


def _get_opcodes(str1: str, str2: str):
    match _STR_CMP:
        case StrCmp.DIFFLIB:
            return difflib.SequenceMatcher(None, str1, str2).get_opcodes()
        case StrCmp.LEVENSHTEIN:
            return Levenshtein.opcodes(str1, str2)


################################################################################


def _get_distance_matrix_from_similarity_matrix(similarity_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # Remove negatives values.
    # We don't care about vector direction, only magnitude (how different are the strings).
    # Similarity vectors from string data rarely contain negative values anyways.
    similarity_matrix = np.abs(similarity_matrix)
    # Calculate distance matirx manually instead of calling cosine_distances because of the previous step.
    # Also remove negatives, just in case, there are cases where we get very small negative values.
    distance_matrix = np.abs(1 - similarity_matrix)
    return distance_matrix


def _get_distance_matrix_tfidf(strings: list[str]) -> npt.NDArray[np.float64]:
    # Vectorize strings.
    vectorizer = TfidfVectorizer()
    document_term_matrix = vectorizer.fit_transform(strings)
    # Compute similarity matrix.
    similarity_matrix = cosine_similarity(document_term_matrix)
    return _get_distance_matrix_from_similarity_matrix(similarity_matrix)


def _get_distance_matrix_ratio(strings: list[str]) -> npt.NDArray[np.float64]:
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
                ratio = _get_ratio(strings[i], strings[j])
                cache[(strings[i], strings[j])] = ratio
            else:
                ratio = cache[(strings[i], strings[j])]
            similarity_matrix[i][j] = ratio
            similarity_matrix[j][i] = ratio  # Mirror the value
    cache.clear()
    return _get_distance_matrix_from_similarity_matrix(similarity_matrix)


################################################################################


def _group_strings_in_clusters(strings: list[str], labels: npt.NDArray) -> list[list[str]]:
    clusters: dict[object, list[str]] = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(strings[i])
    return list(clusters.values())


def cluster_strings_simple(strings: list[str], threshold: float, _unused: None = None) -> list[list[str]]:
    clusters: list[list[str]] = []
    cluster_num = 0
    strings = copy.copy(strings)
    cache: dict[tuple[str, str], float] = {}
    while len(strings) > 0:
        string = strings.pop()
        clusters.append([string])
        indexes_to_remove = []
        for idx, string_to_match in enumerate(strings):
            if (string, string_to_match) not in cache:
                ratio = _get_ratio(string, string_to_match)
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


def cluster_strings_agglo(
    strings: list[str], threshold: float, distance_matrix: npt.NDArray[np.float64]
) -> list[list[str]]:
    clustering = AgglomerativeClustering(
        metric="precomputed", distance_threshold=threshold, linkage="average", n_clusters=None
    )
    labels = clustering.fit_predict(distance_matrix)
    return _group_strings_in_clusters(strings, labels)


def cluster_strings_dbscan(strings: list[str], eps: float, distance_matrix: npt.NDArray[np.float64]) -> list[list[str]]:
    clustering = DBSCAN(metric="precomputed", eps=eps)
    labels = clustering.fit_predict(distance_matrix)
    return _group_strings_in_clusters(strings, labels)


################################################################################


def _globify_strings_internal(strings: list[str]) -> str:
    result = strings[0]
    for string in strings[1:]:
        # Reverse the opcodes so that indexes don't change as the operations are applied.
        # for tag, i1, i2, _j1, _j2 in reversed(difflib.SequenceMatcher(None, result, string).get_opcodes()):
        for tag, i1, i2, _j1, _j2 in reversed(_get_opcodes(result, string)):
            if tag != "equal":
                result = result[:i1] + "*" + result[i2:]
    return result


def globify_strings(strings: list[str]) -> str:
    if len(strings) == 0:
        return ""
    strings = list(set(strings))  # Remove duplicates.
    if len(strings) == 1:
        return strings[0]
    return re.sub(r"\*+", "*", _globify_strings_internal(strings))


def _globify_strings_internal_cached(strings: list[str], cache: dict[tuple[str, str], str]) -> str:
    result = strings[0]
    for string in strings[1:]:
        if (result, string) not in cache:
            result_prev = copy.copy(result)
            # Reverse the opcodes so that indexes don't change as the operations are applied.
            # for tag, i1, i2, _j1, _j2 in reversed(difflib.SequenceMatcher(None, result, string).get_opcodes()):
            for tag, i1, i2, _j1, _j2 in reversed(_get_opcodes(result, string)):
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
    return re.sub(r"\*+", "*", max(results, key=len))


################################################################################


def _find_bound_internal(
    func: Callable[[float], float],
    ref_result: float,
    init_val: float,
    limit_val: float,
    steps: int,
    finetune_loops: int,
    finetune_factor: int,
    ref_result_tol: float,
) -> float:
    step_size = abs(limit_val - init_val) / steps
    step = 0
    result = None
    prev_val = None
    val = init_val
    while True:
        prev_val = val
        val = init_val + step * step_size
        result = func(val)
        if (ref_result - ref_result_tol) <= result <= (ref_result + ref_result_tol):
            finetune_loops -= 1
            if finetune_loops < 1:
                return val
            else:
                if step == 0:
                    return val
                return _find_bound_internal(
                    func, ref_result, prev_val, val, steps, finetune_loops, finetune_factor, ref_result_tol
                )
        step += 1 if limit_val > init_val else -1
        if (step > steps) if limit_val > init_val else (steps < -steps):
            raise RuntimeError("Couldn't find a value within the expected range that satisfies the provided result.")


def find_bounds(
    func: Callable[[float], float],
    ref_result: float,
    lower_init_val: float,
    higher_init_val: float,
    steps: int = 10,
    finetune_loops: int = 2,
    finetune_factor: int = 10,
    ref_result_tol: float = 0,
) -> tuple[float, float]:
    if lower_init_val >= higher_init_val:
        raise ValueError("Lower init val must be less than higher init val.")
    if steps < 2:
        raise ValueError("Steps must be 2 or higher.")
    if finetune_loops < 1:
        raise ValueError("Fine-tune loops value must be 1 or higher.")
    if finetune_factor < 2:
        raise ValueError("Fine-tune factor must be 2 or higher.")
    if ref_result_tol < 0:
        raise ValueError("Ref result tol mus be positive.")

    lower_bound = _find_bound_internal(
        func,
        ref_result,
        lower_init_val,
        higher_init_val,
        steps,
        finetune_loops,
        finetune_factor,
        ref_result_tol,
    )
    print()
    higher_bound = _find_bound_internal(
        func,
        ref_result,
        higher_init_val,
        lower_init_val,
        steps,
        finetune_loops,
        finetune_factor,
        ref_result_tol,
    )
    return lower_bound, higher_bound


################################################################################


def run(
    cluster_string_func: Callable[[list[str], float, Any], list[list[str]]],
    distance_matrix_name: str,
    distance_matrix: npt.NDArray[np.float64] | None,
    steps=10,
    finetune_loops=2,
) -> None:
    print(cluster_string_func.__name__, distance_matrix_name)
    lower_bound, upper_bound = find_bounds(
        lambda param: len(cluster_string_func(strings, param, distance_matrix)),
        num_clusters,
        0.001,
        1.001,
        steps=steps,
        finetune_loops=finetune_loops,
    )
    param = (lower_bound + upper_bound) / 2
    print(f"{lower_bound=}")
    print(f"{upper_bound=}")
    print(f"{param=}")
    start_t = time.time()
    clustered_strings = cluster_string_func(strings, param, distance_matrix)
    print("time taken:", time.time() - start_t)
    print(f"{len(clustered_strings)=}")
    globbed_strings: list[str] = []
    for cluster in clustered_strings:
        globbed_strings.append(globify_strings(cluster))
    globbed_strings.sort()
    print(json.dumps(globbed_strings, indent=2))
    print()


################################################################################


def _generate_random_string(length: int) -> str:
    return "".join(random.choices(ascii_letters + digits, k=length))


def _generate_random_datetime(end: datetime.datetime, delta: datetime.timedelta) -> datetime.datetime:
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return end - datetime.timedelta(seconds=random_seconds)


def _generate_log_messages(num_messages: int) -> list[str]:
    error_types = [
        "SyntaxError: invalid syntax while parsing file",
        "ConnectionError: Failed to establish a new connection",
        "FileNotFoundError: [Errno 2] No such file or directory",
        "PermissionError: [Errno 13] Permission denied",
        "TimeoutError: [Errno 110] Connection timed out",
        "ValueError: invalid literal for int() with base 10",
        "IndexError: list index out of range",
        "KeyError: 'key' is missing from dictionary",
        "AttributeError: 'NoneType' object has no attribute",
        "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
        "OtherError: example of a different error",
        "YetAnotherError: this is also an error bla bla bla foo bar baz",
    ]
    messages = []
    for _ in range(num_messages):
        timestamp = _generate_random_datetime(datetime.datetime.now(), datetime.timedelta(days=50 * 365)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        error = random.choice(error_types)
        # random_text = _generate_random_string(random.randint(0, 5))
        # message = f"{timestamp} {error} {random_text}"
        message = f"{timestamp} {error}"
        messages.append(message)
    return messages


################################################################################


def _find_lines_with_substring(directory: str, file_extension: str, substring: str) -> list[str]:
    matching_lines: list[str] = []
    for filename in os.listdir(directory):
        if filename.endswith(file_extension):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    if substring in line:
                        matching_lines.append(line.strip())
    return matching_lines


################################################################################


if __name__ == "__main__":
    strings = _generate_log_messages(1000)

    num_clusters = 12

    distance_matrices: dict[str, npt.NDArray[np.float64]] = {}
    for str_cmp in StrCmp:
        _STR_CMP = str_cmp
        print(_get_distance_matrix_ratio.__name__)
        start_t = time.time()
        distance_matrices[f"{_get_distance_matrix_ratio.__name__} {str_cmp}"] = _get_distance_matrix_ratio(strings)
        print("time taken:", time.time() - start_t)
        print()
    print(_get_distance_matrix_tfidf.__name__)
    start_t = time.time()
    distance_matrices[_get_distance_matrix_tfidf.__name__] = _get_distance_matrix_tfidf(strings)
    print("time taken:", time.time() - start_t)
    print()

    print("################################################################################\n")

    for str_cmp in StrCmp:
        _STR_CMP = str_cmp
        run(cluster_strings_simple, str(str_cmp), None)

    print("################################################################################\n")

    for distance_matrix_name, distance_matrix in distance_matrices.items():
        run(cluster_strings_agglo, distance_matrix_name, distance_matrix)

    print("################################################################################\n")

    for distance_matrix_name, distance_matrix in distance_matrices.items():
        run(cluster_strings_dbscan, distance_matrix_name, distance_matrix, steps=100, finetune_loops=1)

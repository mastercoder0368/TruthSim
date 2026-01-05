"""Evaluation metrics computation."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .diagnosis_matcher import MatchResult
from .llm_judge import JudgmentResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


def compute_diagnostic_accuracy(
        match_results: List[MatchResult],
) -> Dict[str, float]:
    """
    Compute diagnostic accuracy metrics.

    Args:
        match_results: List of diagnosis match results

    Returns:
        Dictionary with accuracy metrics
    """
    if not match_results:
        return {"top1_accuracy": 0.0, "total": 0}

    matches = sum(1 for r in match_results if r.match)
    total = len(match_results)

    return {
        "top1_accuracy": matches / total,
        "correct": matches,
        "total": total,
    }


def compute_simulation_metrics(
        judgment_results: List[JudgmentResult],
) -> Dict[str, Any]:
    """
    Compute patient simulation quality metrics.

    Args:
        judgment_results: List of LLM Judge results

    Returns:
        Dictionary with simulation quality metrics
    """
    if not judgment_results:
        return {}

    # Truth preservation
    truth_passes = sum(1 for r in judgment_results if r.truth_preservation_pass)
    hallucinations = sum(1 for r in judgment_results if r.hallucination)
    consistency_violations = sum(1 for r in judgment_results if r.consistency_violation)
    diagnosis_leaks = sum(1 for r in judgment_results if r.diagnosis_leak)

    total = len(judgment_results)

    # Realism scores
    realism_scores = [r.average_realism for r in judgment_results if r.average_realism > 0]

    # Utility scores
    utility_scores = [r.average_utility for r in judgment_results if r.average_utility > 0]

    return {
        "truth_preservation": {
            "pass_rate": truth_passes / total,
            "hallucination_rate": hallucinations / total,
            "consistency_violation_rate": consistency_violations / total,
            "diagnosis_leak_rate": diagnosis_leaks / total,
        },
        "realism": {
            "mean": np.mean(realism_scores) if realism_scores else 0,
            "std": np.std(realism_scores) if realism_scores else 0,
            "min": min(realism_scores) if realism_scores else 0,
            "max": max(realism_scores) if realism_scores else 0,
        },
        "clinical_utility": {
            "mean": np.mean(utility_scores) if utility_scores else 0,
            "std": np.std(utility_scores) if utility_scores else 0,
            "min": min(utility_scores) if utility_scores else 0,
            "max": max(utility_scores) if utility_scores else 0,
        },
        "total": total,
    }


def compute_cohens_kappa(
        rater1: List[int],
        rater2: List[int],
) -> float:
    """
    Compute Cohen's Kappa for inter-rater agreement.

    Args:
        rater1: Ratings from first rater
        rater2: Ratings from second rater

    Returns:
        Cohen's Kappa value
    """
    if len(rater1) != len(rater2):
        raise ValueError("Raters must have same number of ratings")

    if len(rater1) == 0:
        return 0.0

    # Get unique categories
    categories = sorted(set(rater1) | set(rater2))
    n = len(rater1)

    # Build confusion matrix
    matrix = np.zeros((len(categories), len(categories)))
    for r1, r2 in zip(rater1, rater2):
        i = categories.index(r1)
        j = categories.index(r2)
        matrix[i, j] += 1

    # Compute observed agreement
    po = np.trace(matrix) / n

    # Compute expected agreement
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    pe = np.sum(row_sums * col_sums) / (n * n)

    # Compute kappa
    if pe == 1:
        return 1.0

    kappa = (po - pe) / (1 - pe)
    return kappa


def compute_pearson_correlation(
        scores1: List[float],
        scores2: List[float],
) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient.

    Args:
        scores1: First set of scores
        scores2: Second set of scores

    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    if len(scores1) != len(scores2):
        raise ValueError("Score lists must have same length")

    if len(scores1) < 2:
        return 0.0, 1.0

    r, p = stats.pearsonr(scores1, scores2)
    return r, p


def compute_agreement(
        human_judgments: List[JudgmentResult],
        llm_judgments: List[JudgmentResult],
) -> Dict[str, Any]:
    """
    Compute agreement between human and LLM evaluations.

    Args:
        human_judgments: Human evaluator results
        llm_judgments: LLM Judge results

    Returns:
        Dictionary with agreement metrics
    """
    if len(human_judgments) != len(llm_judgments):
        raise ValueError("Must have same number of human and LLM judgments")

    results = {}

    # Truth preservation (binary - use Kappa)
    human_truth = [1 if j.truth_preservation_pass else 0 for j in human_judgments]
    llm_truth = [1 if j.truth_preservation_pass else 0 for j in llm_judgments]
    results["truth_preservation_kappa"] = compute_cohens_kappa(human_truth, llm_truth)

    # Realism (continuous - use Pearson)
    human_realism = [j.average_realism for j in human_judgments]
    llm_realism = [j.average_realism for j in llm_judgments]
    r, p = compute_pearson_correlation(human_realism, llm_realism)
    results["realism_correlation"] = r
    results["realism_p_value"] = p

    # Clinical utility (continuous - use Pearson)
    human_utility = [j.average_utility for j in human_judgments]
    llm_utility = [j.average_utility for j in llm_judgments]
    r, p = compute_pearson_correlation(human_utility, llm_utility)
    results["utility_correlation"] = r
    results["utility_p_value"] = p

    return results


def compute_metrics(
        match_results: Optional[List[MatchResult]] = None,
        judgment_results: Optional[List[JudgmentResult]] = None,
        human_judgments: Optional[List[JudgmentResult]] = None,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics.

    Args:
        match_results: Diagnosis matching results
        judgment_results: LLM Judge results
        human_judgments: Human evaluation results (for agreement)

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # Diagnostic accuracy
    if match_results:
        metrics["diagnostic_accuracy"] = compute_diagnostic_accuracy(match_results)

    # Simulation quality
    if judgment_results:
        metrics["simulation_quality"] = compute_simulation_metrics(judgment_results)

    # Human-LLM agreement
    if judgment_results and human_judgments:
        metrics["agreement"] = compute_agreement(human_judgments, judgment_results)

    return metrics


def paired_t_test(
        clean_scores: List[float],
        noisy_scores: List[float],
) -> Dict[str, float]:
    """
    Perform paired t-test for clean vs noisy conditions.

    Args:
        clean_scores: Accuracy scores in clean condition
        noisy_scores: Accuracy scores in noisy condition

    Returns:
        Dictionary with t-statistic, p-value, and effect size
    """
    if len(clean_scores) != len(noisy_scores):
        raise ValueError("Must have same number of scores")

    t_stat, p_value = stats.ttest_rel(clean_scores, noisy_scores)

    # Compute Cohen's d effect size
    diff = np.array(clean_scores) - np.array(noisy_scores)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "mean_difference": np.mean(diff),
        "significant": p_value < 0.001,  # p < 0.001 threshold
    }

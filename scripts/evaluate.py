#!/usr/bin/env python3
"""
Evaluate TruthSim Conversation Transcripts.

This script evaluates conversation transcripts using:
1. Diagnosis matching (semantic comparison with ground truth)
2. LLM-as-Judge evaluation (truth preservation, realism, utility)

Usage:
    python scripts/evaluate.py \
        --conversations data/conversations/ \
        --output data/evaluations/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conversation import Transcript
from src.evaluation import (
    DiagnosisMatcher,
    LLMJudge,
    compute_metrics,
    compute_diagnostic_accuracy,
    compute_simulation_metrics,
)
from src.utils.config import load_config
from src.utils.logger import setup_logger, get_logger


def load_transcripts(input_dir: str) -> List[Transcript]:
    """Load all transcripts from directory (recursive)."""
    transcripts = []
    input_path = Path(input_dir)

    # Search recursively for JSON files
    for filepath in input_path.rglob("*.json"):
        try:
            transcript = Transcript.load(str(filepath))
            transcripts.append(transcript)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")

    return transcripts


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save evaluation results to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TruthSim conversation transcripts"
    )
    parser.add_argument(
        "--conversations", "-c",
        required=True,
        help="Directory containing conversation transcripts"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--matcher-model",
        default="gpt-4o",
        help="Model for diagnosis matching"
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o",
        help="Model for LLM-as-Judge evaluation"
    )
    parser.add_argument(
        "--skip-diagnosis-matching",
        action="store_true",
        help="Skip diagnosis matching evaluation"
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-Judge evaluation"
    )
    parser.add_argument(
        "--max-transcripts",
        type=int,
        default=None,
        help="Maximum transcripts to evaluate (for testing)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger(level=args.log_level)
    logger = get_logger(__name__)

    # Load transcripts
    logger.info(f"Loading transcripts from {args.conversations}")
    transcripts = load_transcripts(args.conversations)

    if args.max_transcripts:
        transcripts = transcripts[:args.max_transcripts]

    logger.info(f"Loaded {len(transcripts)} transcripts")

    if not transcripts:
        logger.error("No transcripts found")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results storage
    all_results = {
        "transcripts_evaluated": len(transcripts),
        "diagnosis_matching": [],
        "llm_judge": [],
        "metrics": {},
    }

    # ========== Diagnosis Matching ==========
    if not args.skip_diagnosis_matching:
        logger.info("Running diagnosis matching...")

        matcher = DiagnosisMatcher(
            model=args.matcher_model,
            provider="openai",
        )

        match_results = []
        for transcript in tqdm(transcripts, desc="Matching diagnoses"):
            if not transcript.final_diagnosis:
                # No diagnosis made
                match_results.append({
                    "conversation_id": transcript.conversation_id,
                    "patient_id": transcript.patient_id,
                    "doctor_diagnosis": None,
                    "ground_truth": transcript.ground_truth_diagnosis,
                    "match": False,
                    "reasoning": "No diagnosis provided",
                })
                continue

            try:
                result = matcher.match(
                    doctor_diagnosis=transcript.final_diagnosis,
                    ground_truth=transcript.ground_truth_diagnosis,
                )

                match_results.append({
                    "conversation_id": transcript.conversation_id,
                    "patient_id": transcript.patient_id,
                    "doctor_diagnosis": transcript.final_diagnosis,
                    "ground_truth": transcript.ground_truth_diagnosis,
                    **result.to_dict(),
                })

            except Exception as e:
                logger.error(f"Error matching {transcript.conversation_id}: {e}")
                match_results.append({
                    "conversation_id": transcript.conversation_id,
                    "patient_id": transcript.patient_id,
                    "error": str(e),
                    "match": False,
                })

        all_results["diagnosis_matching"] = match_results

        # Compute accuracy
        matches = sum(1 for r in match_results if r.get("match", False))
        total = len(match_results)
        accuracy = matches / total if total > 0 else 0

        all_results["metrics"]["diagnostic_accuracy"] = {
            "top1_accuracy": accuracy,
            "correct": matches,
            "total": total,
        }

        logger.info(f"Diagnostic Accuracy: {accuracy:.2%} ({matches}/{total})")

    # ========== LLM-as-Judge Evaluation ==========
    if not args.skip_judge:
        logger.info("Running LLM-as-Judge evaluation...")

        judge = LLMJudge(
            model=args.judge_model,
            provider="openai",
        )

        judge_results = []
        for transcript in tqdm(transcripts, desc="Evaluating with LLM Judge"):
            try:
                result = judge.evaluate(transcript)

                judge_results.append({
                    "conversation_id": transcript.conversation_id,
                    "patient_id": transcript.patient_id,
                    **result.to_dict(),
                })

            except Exception as e:
                logger.error(f"Error evaluating {transcript.conversation_id}: {e}")
                judge_results.append({
                    "conversation_id": transcript.conversation_id,
                    "patient_id": transcript.patient_id,
                    "error": str(e),
                })

        all_results["llm_judge"] = judge_results

        # Compute simulation metrics
        from src.evaluation.llm_judge import JudgmentResult

        valid_results = [
            JudgmentResult.from_llm_response(r.get("section_details", {}))
            for r in judge_results
            if "error" not in r and "section_details" in r
        ]

        if valid_results:
            sim_metrics = compute_simulation_metrics(valid_results)
            all_results["metrics"]["simulation_quality"] = sim_metrics

            logger.info(f"\nSimulation Quality Metrics:")
            logger.info(f"  Truth Preservation Pass Rate: {sim_metrics['truth_preservation']['pass_rate']:.2%}")
            logger.info(f"  Hallucination Rate: {sim_metrics['truth_preservation']['hallucination_rate']:.2%}")
            logger.info(f"  Average Realism: {sim_metrics['realism']['mean']:.2f}/5.0")
            logger.info(f"  Average Utility: {sim_metrics['clinical_utility']['mean']:.2f}/5.0")

    # ========== Save Results ==========

    # Save detailed results
    detailed_path = output_dir / "evaluation_detailed.json"
    save_results(all_results, str(detailed_path))
    logger.info(f"Detailed results saved to {detailed_path}")

    # Save summary
    summary = {
        "transcripts_evaluated": all_results["transcripts_evaluated"],
        "metrics": all_results["metrics"],
    }
    summary_path = output_dir / "evaluation_summary.json"
    save_results(summary, str(summary_path))
    logger.info(f"Summary saved to {summary_path}")

    # ========== Print Summary ==========
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Transcripts evaluated: {len(transcripts)}")

    if "diagnostic_accuracy" in all_results["metrics"]:
        acc = all_results["metrics"]["diagnostic_accuracy"]
        logger.info(f"Diagnostic Accuracy: {acc['top1_accuracy']:.2%}")

    if "simulation_quality" in all_results["metrics"]:
        sq = all_results["metrics"]["simulation_quality"]
        logger.info(f"Truth Preservation: {sq['truth_preservation']['pass_rate']:.2%}")
        logger.info(f"Avg Realism: {sq['realism']['mean']:.2f}")


if __name__ == "__main__":
    main()

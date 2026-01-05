#!/usr/bin/env python3
"""
Run TruthSim Diagnostic Conversations.

This script runs diagnostic conversations between patient simulators
and doctor LLMs, saving transcripts for evaluation.

Usage:
    python scripts/run_simulation.py \
        --patients data/patient_cases/ \
        --umls-cache data/umls_cache/ \
        --doctor-model "meta-llama/Llama-3.1-70B-Instruct" \
        --output data/conversations/
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

from src.simulator import PatientSimulator, create_patient_simulator
from src.doctor import DoctorLLM
from src.conversation import ConversationManager, Transcript
from src.umls import UMLSCache
from src.utils.config import load_config
from src.utils.logger import setup_logger, get_logger


def load_patient_cases(input_dir: str) -> List[Dict[str, Any]]:
    """Load all patient cases from directory."""
    cases = []
    input_path = Path(input_dir)

    for filepath in sorted(input_path.glob("*.json")):
        with open(filepath, "r") as f:
            case = json.load(f)
            if "patient_id" not in case:
                case["patient_id"] = filepath.stem
            cases.append(case)

    return cases


def load_umls_contexts(cache_dir: str, patient_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load UMLS contexts for all patients."""
    cache = UMLSCache(cache_dir)
    contexts = {}

    for patient_id in patient_ids:
        context = cache.load_patient_context(patient_id)
        if context:
            contexts[patient_id] = context

    return contexts


def main():
    parser = argparse.ArgumentParser(
        description="Run TruthSim diagnostic conversations"
    )
    parser.add_argument(
        "--patients", "-p",
        required=True,
        help="Directory containing patient case JSON files"
    )
    parser.add_argument(
        "--umls-cache", "-u",
        required=True,
        help="Directory containing UMLS context cache"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Directory to save conversation transcripts"
    )
    parser.add_argument(
        "--doctor-model", "-d",
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="Doctor LLM model identifier"
    )
    parser.add_argument(
        "--doctor-provider",
        default="together",
        help="Doctor LLM provider (together, openai, anthropic)"
    )
    parser.add_argument(
        "--simulator-model",
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="Patient simulator model identifier"
    )
    parser.add_argument(
        "--simulator-provider",
        default="together",
        help="Patient simulator provider"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=15,
        help="Maximum conversation turns"
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Maximum number of patients to process (for testing)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger(level=args.log_level)
    logger = get_logger(__name__)

    # Load config
    config = load_config(args.config) if args.config else load_config()

    # Load patient cases
    logger.info(f"Loading patient cases from {args.patients}")
    patient_cases = load_patient_cases(args.patients)

    if args.max_patients:
        patient_cases = patient_cases[:args.max_patients]

    logger.info(f"Loaded {len(patient_cases)} patient cases")

    if not patient_cases:
        logger.error("No patient cases found")
        sys.exit(1)

    # Load UMLS contexts
    logger.info(f"Loading UMLS contexts from {args.umls_cache}")
    patient_ids = [c.get("patient_id") for c in patient_cases]
    umls_contexts = load_umls_contexts(args.umls_cache, patient_ids)
    logger.info(f"Loaded contexts for {len(umls_contexts)} patients")

    # Initialize patient simulator
    logger.info(f"Initializing patient simulator: {args.simulator_model}")
    patient_simulator = PatientSimulator(
        model=args.simulator_model,
        provider=args.simulator_provider,
        generator_temperature=config.get("temperature", {}).get("generator", 0.7),
        verifier_temperature=config.get("temperature", {}).get("verifier", 0.0),
        max_regeneration_attempts=config.get("conversation", {}).get("max_regeneration_attempts", 2),
        response_word_limit=config.get("conversation", {}).get("response_word_limit", 50),
    )

    # Initialize doctor LLM
    logger.info(f"Initializing doctor LLM: {args.doctor_model}")
    doctor_llm = DoctorLLM(
        model=args.doctor_model,
        provider=args.doctor_provider,
        temperature=config.get("temperature", {}).get("doctor", 0.3),
    )

    # Initialize conversation manager
    output_dir = Path(args.output) / args.doctor_model.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    manager = ConversationManager(
        patient_simulator=patient_simulator,
        doctor_llm=doctor_llm,
        max_turns=args.max_turns,
        output_dir=str(output_dir),
    )

    # Run conversations
    logger.info("Starting conversations...")
    transcripts = []

    for patient_data in tqdm(patient_cases, desc="Running conversations"):
        patient_id = patient_data.get("patient_id")

        # Get UMLS context
        umls_context = umls_contexts.get(patient_id, {})
        if not umls_context:
            logger.warning(f"No UMLS context for {patient_id}, using empty context")

        try:
            transcript = manager.run_conversation(
                patient_data=patient_data,
                umls_context=umls_context,
                save_transcript=True,
            )
            transcripts.append(transcript)

            # Log result
            status = "✓" if transcript.final_diagnosis else "✗"
            logger.debug(
                f"{status} {patient_id}: "
                f"GT={transcript.ground_truth_diagnosis}, "
                f"Pred={transcript.final_diagnosis}, "
                f"Turns={transcript.get_turn_count()}"
            )

        except Exception as e:
            logger.error(f"Error processing {patient_id}: {e}")
            continue

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("SIMULATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total conversations: {len(transcripts)}")
    logger.info(f"With diagnosis: {sum(1 for t in transcripts if t.final_diagnosis)}")
    logger.info(f"Transcripts saved to: {output_dir}")

    # Print simulator stats
    stats = patient_simulator.get_stats()
    logger.info(f"\nSimulator Statistics:")
    logger.info(f"  Total generations: {stats['total_generations']}")
    logger.info(f"  Regenerations: {stats['regenerations']}")
    logger.info(f"  Pass rate: {stats['pass_rate']:.2%}")


if __name__ == "__main__":
    main()

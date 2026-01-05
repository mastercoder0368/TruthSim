#!/usr/bin/env python3
"""
Batch UMLS Context Extraction Script.

This script preprocesses patient cases by extracting UMLS semantic context
for all symptoms. The extracted contexts are cached for use during simulation.

Usage:
    python scripts/preprocess_umls.py --input data/patient_cases/ --output data/umls_cache/
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.umls import UMLSCache, UMLSContextExtractor
from src.utils.config import load_config, get_api_key
from src.utils.logger import setup_logger, get_logger


def load_patient_cases(input_dir: str) -> list:
    """Load all patient cases from directory."""
    cases = []
    input_path = Path(input_dir)

    for filepath in input_path.glob("*.json"):
        with open(filepath, "r") as f:
            case = json.load(f)
            # Ensure patient_id is set
            if "patient_id" not in case:
                case["patient_id"] = filepath.stem
            cases.append(case)

    return cases


def main():
    parser = argparse.ArgumentParser(
        description="Extract UMLS semantic context for patient symptoms"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Directory containing patient case JSON files"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Directory to save UMLS context cache"
    )
    parser.add_argument(
        "--api-key",
        help="UMLS API key (or set UMLS_API_KEY env var)"
    )
    parser.add_argument(
        "--synonyms-limit",
        type=int,
        default=50,
        help="Maximum synonyms per symptom"
    )
    parser.add_argument(
        "--variations-limit",
        type=int,
        default=25,
        help="Maximum variations per symptom"
    )
    parser.add_argument(
        "--relations-limit",
        type=int,
        default=100,
        help="Maximum SNOMED relations per symptom"
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

    # Get API key
    api_key = args.api_key or os.getenv("UMLS_API_KEY")
    if not api_key:
        logger.error("UMLS API key not provided. Set UMLS_API_KEY or use --api-key")
        sys.exit(1)

    # Load patient cases
    logger.info(f"Loading patient cases from {args.input}")
    patient_cases = load_patient_cases(args.input)
    logger.info(f"Loaded {len(patient_cases)} patient cases")

    if not patient_cases:
        logger.error("No patient cases found")
        sys.exit(1)

    # Initialize cache
    cache = UMLSCache(args.output)

    # Extract contexts
    logger.info("Starting UMLS context extraction...")
    cache.batch_extract(
        patient_cases=patient_cases,
        api_key=api_key,
        synonyms_limit=args.synonyms_limit,
        variations_limit=args.variations_limit,
        relations_limit=args.relations_limit,
    )

    logger.info("UMLS preprocessing complete!")
    logger.info(f"Cached contexts saved to {args.output}")


if __name__ == "__main__":
    main()

"""UMLS context caching utilities."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .extractor import UMLSContextExtractor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class UMLSCache:
    """
    Manager for UMLS context caching.

    Handles saving, loading, and batch processing of UMLS contexts.
    """

    def __init__(self, cache_dir: str):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for storing cached contexts
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Global symptom cache file
        self.global_cache_file = self.cache_dir / "umls_context_cache.json"
        self._global_cache: Optional[Dict[str, Any]] = None

    def _load_global_cache(self) -> Dict[str, Any]:
        """Load the global symptom cache."""
        if self._global_cache is not None:
            return self._global_cache

        if self.global_cache_file.exists():
            with open(self.global_cache_file, "r") as f:
                self._global_cache = json.load(f)
        else:
            self._global_cache = {}

        return self._global_cache

    def _save_global_cache(self) -> None:
        """Save the global symptom cache."""
        if self._global_cache is not None:
            with open(self.global_cache_file, "w") as f:
                json.dump(self._global_cache, f, indent=2)

    def get_symptom_context(self, symptom: str) -> Optional[Dict[str, Any]]:
        """
        Get cached context for a symptom.

        Args:
            symptom: Symptom term

        Returns:
            Cached context or None
        """
        cache = self._load_global_cache()
        key = symptom.lower().replace(" ", "_").replace("-", "_")
        return cache.get(key)

    def add_symptom_context(self, symptom: str, context: Dict[str, Any]) -> None:
        """
        Add a symptom context to the cache.

        Args:
            symptom: Symptom term
            context: Extracted context
        """
        cache = self._load_global_cache()
        key = symptom.lower().replace(" ", "_").replace("-", "_")
        cache[key] = context
        self._global_cache = cache
        self._save_global_cache()

    def get_patient_context_path(self, patient_id: str) -> Path:
        """Get the path for a patient's context file."""
        return self.cache_dir / f"{patient_id}_context.json"

    def load_patient_context(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Load cached context for a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            Patient context or None
        """
        context_path = self.get_patient_context_path(patient_id)

        if context_path.exists():
            with open(context_path, "r") as f:
                return json.load(f)

        return None

    def save_patient_context(
            self,
            patient_id: str,
            context: Dict[str, Any]
    ) -> None:
        """
        Save context for a patient.

        Args:
            patient_id: Patient identifier
            context: Patient context dictionary
        """
        context_path = self.get_patient_context_path(patient_id)

        with open(context_path, "w") as f:
            json.dump(context, f, indent=2)

    def batch_extract(
            self,
            patient_cases: List[Dict[str, Any]],
            api_key: str,
            **kwargs
    ) -> None:
        """
        Batch extract and cache UMLS contexts for all patient cases.

        Args:
            patient_cases: List of patient case dictionaries
            api_key: UMLS API key
            **kwargs: Additional arguments for extractor
        """
        # Collect all unique symptoms
        all_symptoms = set()
        for case in patient_cases:
            symptoms = case.get("symptoms", [])
            all_symptoms.update(symptoms)

        logger.info(f"Extracting context for {len(all_symptoms)} unique symptoms...")

        # Initialize extractor
        extractor = UMLSContextExtractor(api_key, **kwargs)

        # Load existing cache
        cache = self._load_global_cache()

        # Extract context for each unique symptom
        for symptom in tqdm(all_symptoms, desc="Extracting UMLS context"):
            key = symptom.lower().replace(" ", "_").replace("-", "_")

            # Skip if already cached
            if key in cache:
                logger.debug(f"Using cached context for: {symptom}")
                continue

            context = extractor.extract_full_context(symptom)
            if context:
                cache[key] = context

        # Save global cache
        self._global_cache = cache
        self._save_global_cache()

        logger.info(f"Cached {len(cache)} symptom contexts to {self.global_cache_file}")

        # Create per-patient context files
        for case in tqdm(patient_cases, desc="Creating patient contexts"):
            patient_id = case.get("patient_id", f"P{patient_cases.index(case):03d}")
            symptoms = case.get("symptoms", [])

            patient_context = {}
            for symptom in symptoms:
                key = symptom.lower().replace(" ", "_").replace("-", "_")
                if key in cache:
                    patient_context[key] = cache[key]

            self.save_patient_context(patient_id, patient_context)

        logger.info(f"Created {len(patient_cases)} patient context files")


def load_patient_context(patient_id: str, cache_dir: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to load patient context.

    Args:
        patient_id: Patient identifier
        cache_dir: Cache directory path

    Returns:
        Patient context dictionary or None
    """
    cache = UMLSCache(cache_dir)
    return cache.load_patient_context(patient_id)


def save_patient_context(
        patient_id: str,
        context: Dict[str, Any],
        cache_dir: str
) -> None:
    """
    Convenience function to save patient context.

    Args:
        patient_id: Patient identifier
        context: Context dictionary
        cache_dir: Cache directory path
    """
    cache = UMLSCache(cache_dir)
    cache.save_patient_context(patient_id, context)

"""Diagnosis Semantic Matcher using GPT-4o."""

from typing import Any, Dict, Optional

from ..utils.llm_client import LLMClient
from ..utils.config import load_prompt
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MatchResult:
    """Result of diagnosis matching."""

    def __init__(
            self,
            match: bool,
            reasoning: str,
            doctor_diagnosis: str,
            ground_truth: str,
            doctor_normalized: Optional[str] = None,
            ground_truth_normalized: Optional[str] = None,
    ):
        self.match = match
        self.reasoning = reasoning
        self.doctor_diagnosis = doctor_diagnosis
        self.ground_truth = ground_truth
        self.doctor_normalized = doctor_normalized
        self.ground_truth_normalized = ground_truth_normalized

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match": self.match,
            "reasoning": self.reasoning,
            "doctor_diagnosis": self.doctor_diagnosis,
            "ground_truth": self.ground_truth,
            "doctor_normalized": self.doctor_normalized,
            "ground_truth_normalized": self.ground_truth_normalized,
        }


class DiagnosisMatcher:
    """
    Semantic matcher for comparing doctor diagnoses to ground truth.

    Uses GPT-4o to determine if two diagnostic terms refer to the
    same clinical condition, handling abbreviations, synonyms, and
    alternative terminology.
    """

    def __init__(
            self,
            model: str = "gpt-4o",
            provider: str = "openai",
            prompt_template: Optional[str] = None,
            temperature: float = 0.0,
            llm_client: Optional[LLMClient] = None,
    ):
        """
        Initialize the diagnosis matcher.

        Args:
            model: Model to use for matching (default: gpt-4o)
            provider: API provider
            prompt_template: Custom prompt template
            temperature: Sampling temperature (0.0 for deterministic)
            llm_client: Optional pre-configured LLM client
        """
        if llm_client is None:
            self.llm = LLMClient(model=model, provider=provider)
        else:
            self.llm = llm_client

        self.temperature = temperature

        # Load prompt template
        if prompt_template is None:
            self.prompt_template = load_prompt("diagnosis_matcher_prompt")
        else:
            self.prompt_template = prompt_template

    def match(
            self,
            doctor_diagnosis: str,
            ground_truth: str,
    ) -> MatchResult:
        """
        Determine if doctor's diagnosis matches ground truth.

        Args:
            doctor_diagnosis: Diagnosis provided by the doctor LLM
            ground_truth: Ground truth diagnosis from patient case

        Returns:
            MatchResult indicating match/no-match with reasoning
        """
        # Quick exact match check (case-insensitive)
        if doctor_diagnosis.lower().strip() == ground_truth.lower().strip():
            return MatchResult(
                match=True,
                reasoning="Exact match (case-insensitive)",
                doctor_diagnosis=doctor_diagnosis,
                ground_truth=ground_truth,
                doctor_normalized=doctor_diagnosis.lower().strip(),
                ground_truth_normalized=ground_truth.lower().strip(),
            )

        # Use LLM for semantic matching
        prompt = self.prompt_template.replace(
            "{{doctor_diagnosis}}", doctor_diagnosis
        ).replace(
            "{{ground_truth_diagnosis}}", ground_truth
        )

        try:
            result = self.llm.generate_json(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=256,
            )

            return MatchResult(
                match=result.get("match", False),
                reasoning=result.get("reasoning", ""),
                doctor_diagnosis=doctor_diagnosis,
                ground_truth=ground_truth,
                doctor_normalized=result.get("doctor_normalized"),
                ground_truth_normalized=result.get("ground_truth_normalized"),
            )

        except Exception as e:
            logger.error(f"Diagnosis matching failed: {e}")
            # Default to no match on error
            return MatchResult(
                match=False,
                reasoning=f"Matching error: {str(e)}",
                doctor_diagnosis=doctor_diagnosis,
                ground_truth=ground_truth,
            )


def match_diagnosis(
        doctor_diagnosis: str,
        ground_truth: str,
        matcher: Optional[DiagnosisMatcher] = None,
) -> MatchResult:
    """
    Convenience function to match a diagnosis.

    Args:
        doctor_diagnosis: Doctor's diagnosis
        ground_truth: Ground truth diagnosis
        matcher: Optional pre-configured matcher

    Returns:
        MatchResult
    """
    if matcher is None:
        matcher = DiagnosisMatcher()

    return matcher.match(doctor_diagnosis, ground_truth)

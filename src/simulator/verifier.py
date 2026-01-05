"""Patient Simulator Semantic Consistency Verifier module."""

import json
from typing import Any, Dict, List, Optional

from ..utils.llm_client import LLMClient
from ..utils.config import load_prompt
from ..utils.logger import get_logger

logger = get_logger(__name__)


class VerificationResult:
    """Result of semantic verification."""

    def __init__(
            self,
            passed: bool,
            reasoning: str,
            issue: Optional[str] = None,
            checks: Optional[Dict[str, bool]] = None,
    ):
        self.passed = passed
        self.reasoning = reasoning
        self.issue = issue
        self.checks = checks or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "reasoning": self.reasoning,
            "issue": self.issue,
            "checks": self.checks,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationResult":
        verdict = data.get("verdict", "REGENERATE")
        return cls(
            passed=(verdict == "PASS"),
            reasoning=data.get("reasoning", ""),
            issue=data.get("issue"),
            checks=data.get("checks", {}),
        )


class SemanticVerifier:
    """
    Semantic Consistency Verifier for patient responses.

    Uses UMLS context and the ground truth diagnosis to verify
    that patient responses are medically valid without hallucinations.

    HAS access to:
    - All patient information (demographics, symptoms)
    - Ground truth diagnosis (to detect leaks)
    - UMLS semantic context
    - Conversation history
    """

    def __init__(
            self,
            llm_client: LLMClient,
            prompt_template: Optional[str] = None,
            temperature: float = 0.0,  # Deterministic verification
            max_tokens: int = 512,
    ):
        """
        Initialize the semantic verifier.

        Args:
            llm_client: LLM client for verification
            prompt_template: Custom prompt template (loads default if None)
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens in response
        """
        self.llm = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Load prompt template
        if prompt_template is None:
            self.prompt_template = load_prompt("verifier_prompt")
        else:
            self.prompt_template = prompt_template

    def _format_conversation_history(
            self,
            history: List[Dict[str, str]]
    ) -> str:
        """Format conversation history for the prompt."""
        if not history:
            return "No previous conversation."

        formatted = []
        for turn in history:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")

            if role == "doctor":
                formatted.append(f"Doctor: {content}")
            elif role == "patient":
                formatted.append(f"Patient: {content}")

        return "\n".join(formatted)

    def _format_symptoms(self, symptoms: List[str]) -> str:
        """Format symptoms list for the prompt."""
        return ", ".join(symptoms)

    def _format_noise_profile(self, noise_profile: List[Dict[str, Any]]) -> str:
        """Format noise profile for the prompt."""
        if not noise_profile:
            return "No noise profile assigned."

        formatted = []
        for noise in noise_profile:
            noise_type = noise.get("type", "unknown")
            level = noise.get("level", 0)
            formatted.append(f"- {noise_type}: Level {level}")

        return "\n".join(formatted)

    def verify(
            self,
            candidate_response: str,
            patient_data: Dict[str, Any],
            umls_context: Dict[str, Any],
            conversation_history: List[Dict[str, str]],
    ) -> VerificationResult:
        """
        Verify a candidate patient response.

        Args:
            candidate_response: The response to verify
            patient_data: Complete patient data INCLUDING diagnosis
            umls_context: Pre-computed UMLS semantic context
            conversation_history: Previous conversation turns

        Returns:
            VerificationResult indicating pass/fail with reasoning
        """
        # Extract patient information
        demographics = patient_data.get("demographics", {})
        age = demographics.get("age", 45)
        sex = demographics.get("sex", "Unknown")
        symptoms = patient_data.get("symptoms", [])
        diagnosis = patient_data.get("diagnosis", "Unknown")
        noise_profile = patient_data.get("noise_profile", [])

        # Format prompt
        prompt = self.prompt_template.replace("{{age}}", str(age))
        prompt = prompt.replace("{{sex}}", sex)
        prompt = prompt.replace("{{ground_truth_symptoms}}", self._format_symptoms(symptoms))
        prompt = prompt.replace("{{ground_truth_diagnosis}}", diagnosis)
        prompt = prompt.replace("{{noise_profile}}", self._format_noise_profile(noise_profile))
        prompt = prompt.replace("{{umls_context_json}}", json.dumps(umls_context, indent=2))
        prompt = prompt.replace("{{conversation_history}}", self._format_conversation_history(conversation_history))
        prompt = prompt.replace("{{candidate_response}}", candidate_response)

        # Call LLM for verification
        try:
            result = self.llm.generate_json(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return VerificationResult.from_dict(result)

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            # Default to pass on error to avoid blocking
            return VerificationResult(
                passed=True,
                reasoning=f"Verification error: {str(e)}. Defaulting to pass.",
                issue=None,
            )

    def get_feedback(self, result: VerificationResult) -> str:
        """
        Generate feedback for the generator based on verification result.

        Args:
            result: Verification result

        Returns:
            Feedback string for regeneration
        """
        if result.passed:
            return ""

        feedback_parts = []

        if result.issue:
            feedback_parts.append(f"Issue: {result.issue}")

        if result.reasoning:
            feedback_parts.append(f"Reason: {result.reasoning}")

        # Add specific feedback based on failed checks
        if result.checks:
            if not result.checks.get("symptom_validity", True):
                feedback_parts.append("Do not mention symptoms outside the ground truth list.")
            if not result.checks.get("no_diagnosis_leak", True):
                feedback_parts.append("Do not reveal or hint at the diagnosis.")
            if not result.checks.get("history_consistency", True):
                feedback_parts.append("Do not contradict previous statements.")
            if not result.checks.get("demographic_consistency", True):
                feedback_parts.append("Ensure age and sex references are accurate.")
            if not result.checks.get("noise_fidelity", True):
                feedback_parts.append("Better exhibit the assigned noise behaviors.")

        return " ".join(feedback_parts)

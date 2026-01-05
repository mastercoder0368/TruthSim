"""LLM-as-a-Judge for evaluating patient simulation quality."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.llm_client import LLMClient
from ..utils.config import load_prompt
from ..utils.logger import get_logger
from ..conversation.transcript import Transcript

logger = get_logger(__name__)


@dataclass
class JudgmentResult:
    """Result of LLM-as-Judge evaluation."""

    # Section A: Truth Preservation
    hallucination: bool = False
    consistency_violation: bool = False
    diagnosis_leak: bool = False
    truth_preservation_pass: bool = True

    # Section B: Realism (1-5 scores)
    natural_language: float = 0.0
    noise_behavior: float = 0.0
    personality_consistency: float = 0.0
    authentic_language: float = 0.0
    disclosure_pattern: float = 0.0
    average_realism: float = 0.0

    # Section C: Clinical Utility (1-5 scores)
    history_difficulty: float = 0.0
    diagnosability: float = 0.0
    training_value: float = 0.0
    average_utility: float = 0.0

    # Reasoning
    overall_reasoning: str = ""
    section_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "truth_preservation": {
                "hallucination": self.hallucination,
                "consistency_violation": self.consistency_violation,
                "diagnosis_leak": self.diagnosis_leak,
                "pass": self.truth_preservation_pass,
            },
            "realism": {
                "natural_language": self.natural_language,
                "noise_behavior": self.noise_behavior,
                "personality_consistency": self.personality_consistency,
                "authentic_language": self.authentic_language,
                "disclosure_pattern": self.disclosure_pattern,
                "average": self.average_realism,
            },
            "clinical_utility": {
                "history_difficulty": self.history_difficulty,
                "diagnosability": self.diagnosability,
                "training_value": self.training_value,
                "average": self.average_utility,
            },
            "overall_reasoning": self.overall_reasoning,
            "section_details": self.section_details,
        }

    @classmethod
    def from_llm_response(cls, response: Dict[str, Any]) -> "JudgmentResult":
        """Create JudgmentResult from LLM response."""
        result = cls()

        # Parse Section A
        section_a = response.get("section_a", {})
        result.hallucination = section_a.get("A1_hallucination", {}).get("answer", "No") == "Yes"
        result.consistency_violation = section_a.get("A2_consistency", {}).get("answer", "No") == "Yes"
        result.diagnosis_leak = section_a.get("A3_diagnosis_leak", {}).get("answer", "No") == "Yes"
        result.truth_preservation_pass = not (
                result.hallucination or result.consistency_violation or result.diagnosis_leak
        )

        # Parse Section B
        section_b = response.get("section_b", {})
        result.natural_language = section_b.get("B1_natural_language", {}).get("score", 0)
        result.noise_behavior = section_b.get("B2_noise_behavior", {}).get("score", 0)
        result.personality_consistency = section_b.get("B3_personality_consistency", {}).get("score", 0)
        result.authentic_language = section_b.get("B4_authentic_language", {}).get("score", 0)
        result.disclosure_pattern = section_b.get("B5_disclosure_pattern", {}).get("score", 0)

        realism_scores = [
            result.natural_language,
            result.noise_behavior,
            result.personality_consistency,
            result.authentic_language,
            result.disclosure_pattern,
        ]
        result.average_realism = sum(realism_scores) / len(realism_scores) if realism_scores else 0

        # Parse Section C
        section_c = response.get("section_c", {})
        result.history_difficulty = section_c.get("C1_history_difficulty", {}).get("score", 0)
        result.diagnosability = section_c.get("C2_diagnosability", {}).get("score", 0)
        result.training_value = section_c.get("C3_training_value", {}).get("score", 0)

        utility_scores = [
            result.history_difficulty,
            result.diagnosability,
            result.training_value,
        ]
        result.average_utility = sum(utility_scores) / len(utility_scores) if utility_scores else 0

        # Overall reasoning
        result.overall_reasoning = response.get("overall_reasoning", "")
        result.section_details = response

        return result


class LLMJudge:
    """
    LLM-as-a-Judge for evaluating patient simulation quality.

    Evaluates transcripts on:
    - Section A: Truth Preservation (Yes/No)
    - Section B: Realism (1-5 scale)
    - Section C: Clinical Utility (1-5 scale)
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
        Initialize the LLM Judge.

        Args:
            model: Model to use for evaluation (default: gpt-4o)
            provider: API provider
            prompt_template: Custom prompt template
            temperature: Sampling temperature (0.0 for consistent scoring)
            llm_client: Optional pre-configured LLM client
        """
        if llm_client is None:
            self.llm = LLMClient(model=model, provider=provider)
        else:
            self.llm = llm_client

        self.temperature = temperature

        # Load prompt template
        if prompt_template is None:
            self.prompt_template = load_prompt("judge_prompt")
        else:
            self.prompt_template = prompt_template

    def _format_transcript(self, transcript: Transcript) -> str:
        """Format transcript for the prompt."""
        lines = []
        for turn in transcript.turns:
            role = "Doctor" if turn.role == "doctor" else "Patient"
            lines.append(f"[Turn {turn.turn_number}] {role}: {turn.content}")
        return "\n".join(lines)

    def _format_symptoms(self, symptoms: List[str]) -> str:
        """Format symptoms list."""
        return ", ".join(symptoms)

    def _format_noise_profile(self, noise_profile: List[Dict[str, Any]]) -> str:
        """Format noise profile."""
        if not noise_profile:
            return "No noise profile assigned."

        formatted = []
        for noise in noise_profile:
            formatted.append(f"- {noise.get('type', 'unknown')}: Level {noise.get('level', 0)}")
        return "\n".join(formatted)

    def evaluate(self, transcript: Transcript) -> JudgmentResult:
        """
        Evaluate a conversation transcript.

        Args:
            transcript: Completed conversation transcript

        Returns:
            JudgmentResult with scores and reasoning
        """
        patient_data = transcript.patient_data
        symptoms = patient_data.get("symptoms", [])
        diagnosis = patient_data.get("diagnosis", transcript.ground_truth_diagnosis or "Unknown")
        noise_profile = transcript.noise_profile

        # Format prompt
        prompt = self.prompt_template.replace(
            "{{ground_truth_symptoms}}", self._format_symptoms(symptoms)
        ).replace(
            "{{ground_truth_diagnosis}}", diagnosis
        ).replace(
            "{{noise_profile}}", self._format_noise_profile(noise_profile)
        ).replace(
            "{{transcript}}", self._format_transcript(transcript)
        )

        try:
            response = self.llm.generate_json(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=2048,
            )

            return JudgmentResult.from_llm_response(response)

        except Exception as e:
            logger.error(f"LLM Judge evaluation failed: {e}")
            # Return default result on error
            return JudgmentResult(
                overall_reasoning=f"Evaluation error: {str(e)}",
            )

    def evaluate_batch(
            self,
            transcripts: List[Transcript],
    ) -> List[JudgmentResult]:
        """
        Evaluate multiple transcripts.

        Args:
            transcripts: List of transcripts to evaluate

        Returns:
            List of JudgmentResults
        """
        results = []

        for transcript in transcripts:
            try:
                result = self.evaluate(transcript)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating {transcript.conversation_id}: {e}")
                results.append(JudgmentResult())

        return results

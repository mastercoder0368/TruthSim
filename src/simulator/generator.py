"""Patient Simulator Response Generator module."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..utils.llm_client import LLMClient
from ..utils.config import load_prompt, load_noise_behaviors, find_project_root
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ResponseGenerator:
    """
    Patient Simulator Response Generator.

    Generates patient responses with configurable noise behaviors.
    Does NOT have access to the final diagnosis to prevent data leakage.
    """

    def __init__(
            self,
            llm_client: LLMClient,
            prompt_template: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 100,
            word_limit: int = 50,
    ):
        """
        Initialize the response generator.

        Args:
            llm_client: LLM client for generation
            prompt_template: Custom prompt template (loads default if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            word_limit: Word limit for patient responses
        """
        self.llm = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.word_limit = word_limit

        # Load prompt template
        if prompt_template is None:
            self.prompt_template = load_prompt("generator_prompt")
        else:
            self.prompt_template = prompt_template

        # Load noise behaviors
        self.noise_behaviors = load_noise_behaviors()

    def _get_noise_behavior(self, noise_type: str, level: int) -> str:
        """Get the behavior description for a noise type and level."""
        if noise_type not in self.noise_behaviors:
            logger.warning(f"Unknown noise type: {noise_type}")
            return ""

        behaviors = self.noise_behaviors[noise_type]["levels"]
        level = max(0, min(4, level))  # Clamp to 0-4

        return behaviors.get(level, behaviors.get(str(level), ""))

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

    def generate(
            self,
            patient_data: Dict[str, Any],
            conversation_history: List[Dict[str, str]],
            doctor_question: str,
            feedback: Optional[str] = None,
    ) -> str:
        """
        Generate a patient response.

        Args:
            patient_data: Patient case data (demographics, symptoms, noise profile)
                         NOTE: Should NOT contain diagnosis
            conversation_history: Previous conversation turns
            doctor_question: Current question from doctor
            feedback: Optional feedback from verifier for regeneration

        Returns:
            Generated patient response
        """
        # Extract patient information
        demographics = patient_data.get("demographics", {})
        age = demographics.get("age", 45)
        sex = demographics.get("sex", "Unknown")
        symptoms = patient_data.get("symptoms", [])
        noise_profile = patient_data.get("noise_profile", [])

        # Get noise behaviors
        noise_1 = noise_profile[0] if len(noise_profile) > 0 else {"type": "memory", "level": 0}
        noise_2 = noise_profile[1] if len(noise_profile) > 1 else {"type": "health_literacy", "level": 0}

        behavior_1 = self._get_noise_behavior(noise_1["type"], noise_1["level"])
        behavior_2 = self._get_noise_behavior(noise_2["type"], noise_2["level"])

        noise_name_1 = self.noise_behaviors.get(noise_1["type"], {}).get("name", noise_1["type"])
        noise_name_2 = self.noise_behaviors.get(noise_2["type"], {}).get("name", noise_2["type"])

        # Format prompt
        prompt = self.prompt_template.replace("{{age}}", str(age))
        prompt = prompt.replace("{{sex}}", sex)
        prompt = prompt.replace("{{symptoms_list}}", self._format_symptoms(symptoms))
        prompt = prompt.replace("{{diagnosis_label}}", "[HIDDEN]")  # Never reveal diagnosis
        prompt = prompt.replace("{{noise_type_1}}", noise_name_1)
        prompt = prompt.replace("{{level_1}}", str(noise_1["level"]))
        prompt = prompt.replace("{{behavior_1}}", behavior_1)
        prompt = prompt.replace("{{noise_type_2}}", noise_name_2)
        prompt = prompt.replace("{{level_2}}", str(noise_2["level"]))
        prompt = prompt.replace("{{behavior_2}}", behavior_2)
        prompt = prompt.replace("{{conversation_history}}", self._format_conversation_history(conversation_history))
        prompt = prompt.replace("{{doctor_question}}", doctor_question)

        # Add feedback if provided (for regeneration)
        if feedback:
            prompt += f"\n\n[REGENERATION FEEDBACK]\nYour previous response was rejected: {feedback}\nPlease generate a new response addressing this issue."

        # Generate response
        response = self.llm.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Clean up response
        response = response.strip()

        # Enforce word limit
        words = response.split()
        if len(words) > self.word_limit:
            response = " ".join(words[:self.word_limit]) + "..."

        return response

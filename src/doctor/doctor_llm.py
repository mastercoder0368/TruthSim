"""Doctor LLM module for diagnostic conversations."""

import re
from typing import Any, Dict, List, Optional, Tuple

from ..utils.llm_client import LLMClient
from ..utils.config import load_prompt
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DoctorResponse:
    """Structured doctor response."""

    def __init__(
            self,
            content: str,
            is_diagnosis: bool = False,
            diagnosis: Optional[str] = None,
    ):
        self.content = content
        self.is_diagnosis = is_diagnosis
        self.diagnosis = diagnosis

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "is_diagnosis": self.is_diagnosis,
            "diagnosis": self.diagnosis,
        }


class DoctorLLM:
    """
    Doctor LLM for conducting diagnostic interviews.

    The doctor asks questions to gather clinical information and
    eventually provides a final diagnosis.
    """

    # Pattern to detect final diagnosis
    DIAGNOSIS_PATTERN = re.compile(
        r"final\s*diagnosis[:\s]+(.+?)(?:\.|$)",
        re.IGNORECASE | re.DOTALL
    )

    def __init__(
            self,
            model: str,
            provider: str = "together",
            prompt_template: Optional[str] = None,
            temperature: float = 0.3,
            max_tokens: int = 256,
            llm_client: Optional[LLMClient] = None,
    ):
        """
        Initialize the Doctor LLM.

        Args:
            model: Model identifier
            provider: API provider
            prompt_template: Custom prompt template
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            llm_client: Optional pre-configured LLM client
        """
        if llm_client is None:
            self.llm = LLMClient(model=model, provider=provider)
        else:
            self.llm = llm_client

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Load prompt template
        if prompt_template is None:
            self.prompt_template = load_prompt("doctor_prompt")
        else:
            self.prompt_template = prompt_template

    def _format_conversation_history(
            self,
            history: List[Dict[str, str]]
    ) -> str:
        """Format conversation history for the prompt."""
        if not history:
            return "This is the start of the conversation. Begin by asking about the patient's chief complaint."

        formatted = []
        for turn in history:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")

            if role == "doctor":
                formatted.append(f"Doctor: {content}")
            elif role == "patient":
                formatted.append(f"Patient: {content}")

        return "\n".join(formatted)

    def _parse_response(self, response: str) -> DoctorResponse:
        """
        Parse doctor response to detect if it contains a final diagnosis.

        Args:
            response: Raw response from LLM

        Returns:
            DoctorResponse with parsed information
        """
        response = response.strip()

        # Check for final diagnosis pattern
        match = self.DIAGNOSIS_PATTERN.search(response)

        if match:
            diagnosis = match.group(1).strip()
            # Clean up the diagnosis
            diagnosis = diagnosis.rstrip(".,;")

            return DoctorResponse(
                content=response,
                is_diagnosis=True,
                diagnosis=diagnosis,
            )

        # Also check for alternative patterns
        alt_patterns = [
            r"(?:my|the)\s*diagnosis\s*(?:is|:)\s*(.+?)(?:\.|$)",
            r"i\s*(?:believe|think|suspect)\s*(?:this is|you have)\s*(.+?)(?:\.|$)",
            r"(?:you have|this is)\s*(.+?)(?:\.|$)",
        ]

        for pattern in alt_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match and "final" in response.lower():
                diagnosis = match.group(1).strip().rstrip(".,;")
                return DoctorResponse(
                    content=response,
                    is_diagnosis=True,
                    diagnosis=diagnosis,
                )

        return DoctorResponse(
            content=response,
            is_diagnosis=False,
            diagnosis=None,
        )

    def respond(
            self,
            conversation_history: List[Dict[str, str]],
    ) -> DoctorResponse:
        """
        Generate a doctor response based on conversation history.

        Args:
            conversation_history: Previous conversation turns

        Returns:
            DoctorResponse with content and diagnosis info
        """
        # Format prompt
        prompt = self.prompt_template.replace(
            "{{conversation_history}}",
            self._format_conversation_history(conversation_history)
        )

        # Generate response
        response = self.llm.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Parse and return
        return self._parse_response(response)

    def get_initial_question(self) -> str:
        """Get the doctor's initial question to start the conversation."""
        return "Hello, I'm your doctor today. What brings you in? Can you tell me about your main concern?"


def create_doctor_llm(
        model: str,
        provider: str = "together",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
) -> DoctorLLM:
    """
    Factory function to create a DoctorLLM.

    Args:
        model: Model identifier
        provider: API provider
        config: Configuration dictionary
        **kwargs: Override configuration values

    Returns:
        Configured DoctorLLM instance
    """
    if config is None:
        config = {}

    temp_config = config.get("temperature", {})

    return DoctorLLM(
        model=model,
        provider=provider,
        temperature=kwargs.get("temperature", temp_config.get("doctor", 0.3)),
        max_tokens=kwargs.get("max_tokens", config.get("tokens", {}).get("max_output", 256)),
    )

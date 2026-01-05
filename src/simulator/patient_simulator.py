"""
Patient Simulator with Dual-LLM Architecture.

Combines the Response Generator and Semantic Verifier into a unified
patient simulation system with Generate-Verify-Refine loop.
"""

from typing import Any, Dict, List, Optional, Tuple

from .generator import ResponseGenerator
from .verifier import SemanticVerifier, VerificationResult
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PatientSimulator:
    """
    Truth-Preserving Patient Simulator.

    Implements the Dual-LLM architecture:
    1. Generator: Creates patient responses with noise (no diagnosis access)
    2. Verifier: Validates responses against UMLS context (has diagnosis access)

    The Generate-Verify-Refine loop ensures responses are both realistic
    and medically valid.
    """

    def __init__(
            self,
            model: str = "meta-llama/Llama-3.1-70B-Instruct",
            provider: str = "together",
            generator_temperature: float = 0.7,
            verifier_temperature: float = 0.0,
            max_regeneration_attempts: int = 2,
            response_word_limit: int = 50,
            llm_client: Optional[LLMClient] = None,
    ):
        """
        Initialize the Patient Simulator.

        Args:
            model: Model identifier for both generator and verifier
            provider: API provider
            generator_temperature: Temperature for response generation
            verifier_temperature: Temperature for verification (0.0 = deterministic)
            max_regeneration_attempts: Max retries when verification fails
            response_word_limit: Word limit for patient responses
            llm_client: Optional pre-configured LLM client
        """
        # Create LLM client if not provided
        if llm_client is None:
            llm_client = LLMClient(model=model, provider=provider)

        self.llm = llm_client
        self.max_regeneration_attempts = max_regeneration_attempts

        # Initialize generator and verifier
        self.generator = ResponseGenerator(
            llm_client=llm_client,
            temperature=generator_temperature,
            word_limit=response_word_limit,
        )

        self.verifier = SemanticVerifier(
            llm_client=llm_client,
            temperature=verifier_temperature,
        )

        # Statistics
        self.stats = {
            "total_generations": 0,
            "total_verifications": 0,
            "regenerations": 0,
            "passes": 0,
            "failures": 0,
        }

    def _create_generator_patient_data(
            self,
            patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create patient data for the generator (WITHOUT diagnosis).

        Args:
            patient_data: Complete patient data

        Returns:
            Patient data without diagnosis
        """
        return {
            "patient_id": patient_data.get("patient_id"),
            "demographics": patient_data.get("demographics", {}),
            "symptoms": patient_data.get("symptoms", []),
            "noise_profile": patient_data.get("noise_profile", []),
            # NOTE: diagnosis is NOT included
        }

    def respond(
            self,
            patient_data: Dict[str, Any],
            umls_context: Dict[str, Any],
            conversation_history: List[Dict[str, str]],
            doctor_question: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a verified patient response using Generate-Verify-Refine loop.

        Args:
            patient_data: Complete patient data INCLUDING diagnosis
            umls_context: Pre-computed UMLS semantic context
            conversation_history: Previous conversation turns
            doctor_question: Current question from doctor

        Returns:
            Tuple of (response, metadata)
        """
        # Create data for generator (no diagnosis)
        generator_data = self._create_generator_patient_data(patient_data)

        # Initialize metadata
        metadata = {
            "attempts": 0,
            "verification_results": [],
            "final_status": "unknown",
        }

        feedback = None

        for attempt in range(self.max_regeneration_attempts + 1):
            metadata["attempts"] = attempt + 1
            self.stats["total_generations"] += 1

            # Generate response
            logger.debug(f"Generation attempt {attempt + 1}")
            response = self.generator.generate(
                patient_data=generator_data,
                conversation_history=conversation_history,
                doctor_question=doctor_question,
                feedback=feedback,
            )

            # Verify response
            self.stats["total_verifications"] += 1
            result = self.verifier.verify(
                candidate_response=response,
                patient_data=patient_data,  # Full data with diagnosis
                umls_context=umls_context,
                conversation_history=conversation_history,
            )

            metadata["verification_results"].append(result.to_dict())

            if result.passed:
                logger.debug(f"Response passed verification on attempt {attempt + 1}")
                self.stats["passes"] += 1
                metadata["final_status"] = "passed"
                return response, metadata

            # Verification failed
            logger.debug(f"Response failed verification: {result.issue}")
            self.stats["regenerations"] += 1

            # Get feedback for next attempt
            feedback = self.verifier.get_feedback(result)

            if attempt < self.max_regeneration_attempts:
                logger.debug(f"Regenerating with feedback: {feedback[:100]}...")

        # Max attempts reached - return last response with warning
        logger.warning(
            f"Max regeneration attempts ({self.max_regeneration_attempts}) reached. "
            f"Returning last response despite verification failure."
        )
        self.stats["failures"] += 1
        metadata["final_status"] = "max_attempts_reached"

        return response, metadata

    def get_stats(self) -> Dict[str, Any]:
        """Get generation/verification statistics."""
        stats = self.stats.copy()

        # Calculate rates
        if stats["total_verifications"] > 0:
            stats["pass_rate"] = stats["passes"] / stats["total_verifications"]
            stats["regeneration_rate"] = stats["regenerations"] / stats["total_verifications"]
        else:
            stats["pass_rate"] = 0.0
            stats["regeneration_rate"] = 0.0

        return stats

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            "total_generations": 0,
            "total_verifications": 0,
            "regenerations": 0,
            "passes": 0,
            "failures": 0,
        }


def create_patient_simulator(
        config: Optional[Dict[str, Any]] = None,
        **kwargs
) -> PatientSimulator:
    """
    Factory function to create a PatientSimulator from config.

    Args:
        config: Configuration dictionary
        **kwargs: Override configuration values

    Returns:
        Configured PatientSimulator instance
    """
    if config is None:
        config = {}

    # Extract configuration
    model = kwargs.get(
        "model",
        config.get("models", {}).get("patient_simulator", {}).get(
            "model_name", "meta-llama/Llama-3.1-70B-Instruct"
        )
    )

    provider = kwargs.get(
        "provider",
        config.get("models", {}).get("patient_simulator", {}).get(
            "provider", "together"
        )
    )

    temp_config = config.get("temperature", {})
    conversation_config = config.get("conversation", {})

    return PatientSimulator(
        model=model,
        provider=provider,
        generator_temperature=kwargs.get("generator_temperature", temp_config.get("generator", 0.7)),
        verifier_temperature=kwargs.get("verifier_temperature", temp_config.get("verifier", 0.0)),
        max_regeneration_attempts=kwargs.get(
            "max_regeneration_attempts",
            conversation_config.get("max_regeneration_attempts", 2)
        ),
        response_word_limit=kwargs.get(
            "response_word_limit",
            conversation_config.get("response_word_limit", 50)
        ),
    )

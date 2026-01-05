"""Conversation Manager for running diagnostic conversations."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .transcript import Transcript, create_transcript
from ..simulator import PatientSimulator
from ..doctor import DoctorLLM
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ConversationManager:
    """
    Manages the conversation loop between patient simulator and doctor LLM.

    Orchestrates:
    1. Initial doctor question
    2. Patient response generation with verification
    3. Doctor follow-up or diagnosis
    4. Loop until diagnosis or max turns
    """

    def __init__(
            self,
            patient_simulator: PatientSimulator,
            doctor_llm: DoctorLLM,
            max_turns: int = 15,
            output_dir: Optional[str] = None,
    ):
        """
        Initialize the conversation manager.

        Args:
            patient_simulator: Patient simulator instance
            doctor_llm: Doctor LLM instance
            max_turns: Maximum conversation turns
            output_dir: Directory to save transcripts
        """
        self.patient_simulator = patient_simulator
        self.doctor = doctor_llm
        self.max_turns = max_turns
        self.output_dir = Path(output_dir) if output_dir else None

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_conversation(
            self,
            patient_data: Dict[str, Any],
            umls_context: Dict[str, Any],
            save_transcript: bool = True,
    ) -> Transcript:
        """
        Run a complete diagnostic conversation.

        Args:
            patient_data: Patient case data
            umls_context: Pre-computed UMLS semantic context
            save_transcript: Whether to save transcript to file

        Returns:
            Completed Transcript
        """
        patient_id = patient_data.get("patient_id", "unknown")
        logger.info(f"Starting conversation for patient {patient_id}")

        # Create transcript
        transcript = create_transcript(
            patient_data=patient_data,
            doctor_model=self.doctor.model,
        )

        # Start conversation with doctor's initial question
        initial_question = self.doctor.get_initial_question()
        transcript.add_doctor_turn(initial_question)

        logger.debug(f"Doctor: {initial_question}")

        # Conversation loop
        turn_count = 1
        diagnosis_reached = False

        while turn_count <= self.max_turns:
            # Get patient response
            history = transcript.get_history()
            last_doctor_message = history[-1]["content"] if history else initial_question

            patient_response, response_metadata = self.patient_simulator.respond(
                patient_data=patient_data,
                umls_context=umls_context,
                conversation_history=history[:-1] if history else [],  # Exclude last doctor message
                doctor_question=last_doctor_message,
            )

            transcript.add_patient_turn(
                content=patient_response,
                verification=response_metadata,
            )

            logger.debug(f"Patient: {patient_response}")

            # Get doctor response
            doctor_response = self.doctor.respond(
                conversation_history=transcript.get_history(),
            )

            transcript.add_doctor_turn(
                content=doctor_response.content,
                is_diagnosis=doctor_response.is_diagnosis,
            )

            logger.debug(f"Doctor: {doctor_response.content}")

            # Check if diagnosis reached
            if doctor_response.is_diagnosis:
                logger.info(f"Diagnosis reached: {doctor_response.diagnosis}")
                transcript.finalize(
                    final_diagnosis=doctor_response.diagnosis,
                    metadata={"diagnosis_turn": turn_count + 1},
                )
                diagnosis_reached = True
                break

            turn_count += 1

        # Handle max turns reached
        if not diagnosis_reached:
            logger.warning(f"Max turns ({self.max_turns}) reached without diagnosis")
            transcript.finalize(
                metadata={"terminated_reason": "max_turns_reached"},
            )

        # Save transcript
        if save_transcript and self.output_dir:
            filepath = self.output_dir / f"{transcript.conversation_id}.json"
            transcript.save(str(filepath))
            logger.info(f"Transcript saved to {filepath}")

        return transcript

    def run_batch(
            self,
            patient_cases: List[Dict[str, Any]],
            umls_contexts: Dict[str, Dict[str, Any]],
            save_transcripts: bool = True,
    ) -> List[Transcript]:
        """
        Run conversations for multiple patient cases.

        Args:
            patient_cases: List of patient case data
            umls_contexts: Dictionary mapping patient_id to UMLS context
            save_transcripts: Whether to save transcripts

        Returns:
            List of completed Transcripts
        """
        transcripts = []

        for i, patient_data in enumerate(patient_cases):
            patient_id = patient_data.get("patient_id", f"P{i:03d}")

            # Get UMLS context for this patient
            umls_context = umls_contexts.get(patient_id, {})

            if not umls_context:
                logger.warning(f"No UMLS context found for {patient_id}")

            try:
                transcript = self.run_conversation(
                    patient_data=patient_data,
                    umls_context=umls_context,
                    save_transcript=save_transcripts,
                )
                transcripts.append(transcript)

            except Exception as e:
                logger.error(f"Error processing patient {patient_id}: {e}")
                continue

        return transcripts


def run_single_conversation(
        patient_data: Dict[str, Any],
        umls_context: Dict[str, Any],
        patient_simulator: PatientSimulator,
        doctor_llm: DoctorLLM,
        max_turns: int = 15,
        output_dir: Optional[str] = None,
) -> Transcript:
    """
    Convenience function to run a single conversation.

    Args:
        patient_data: Patient case data
        umls_context: UMLS semantic context
        patient_simulator: Patient simulator instance
        doctor_llm: Doctor LLM instance
        max_turns: Maximum turns
        output_dir: Optional output directory

    Returns:
        Completed Transcript
    """
    manager = ConversationManager(
        patient_simulator=patient_simulator,
        doctor_llm=doctor_llm,
        max_turns=max_turns,
        output_dir=output_dir,
    )

    return manager.run_conversation(
        patient_data=patient_data,
        umls_context=umls_context,
    )

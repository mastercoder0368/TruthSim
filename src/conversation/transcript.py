"""Conversation transcript data structures and utilities."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Turn:
    """A single turn in the conversation."""

    role: str  # "doctor" or "patient"
    content: str
    turn_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Turn":
        return cls(**data)


@dataclass
class Transcript:
    """Complete conversation transcript."""

    conversation_id: str
    patient_id: str
    doctor_model: str
    patient_data: Dict[str, Any]
    turns: List[Turn] = field(default_factory=list)
    final_diagnosis: Optional[str] = None
    ground_truth_diagnosis: Optional[str] = None
    noise_profile: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now().isoformat()

    def add_turn(
            self,
            role: str,
            content: str,
            metadata: Optional[Dict[str, Any]] = None,
    ) -> Turn:
        """Add a turn to the conversation."""
        turn = Turn(
            role=role,
            content=content,
            turn_number=len(self.turns) + 1,
            metadata=metadata or {},
        )
        self.turns.append(turn)
        return turn

    def add_doctor_turn(self, content: str, **kwargs) -> Turn:
        """Add a doctor turn."""
        return self.add_turn("doctor", content, kwargs)

    def add_patient_turn(self, content: str, **kwargs) -> Turn:
        """Add a patient turn."""
        return self.add_turn("patient", content, kwargs)

    def finalize(
            self,
            final_diagnosis: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Finalize the transcript."""
        self.end_time = datetime.now().isoformat()
        if final_diagnosis:
            self.final_diagnosis = final_diagnosis
        if metadata:
            self.metadata.update(metadata)

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history in simple format for prompts."""
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def get_turn_count(self) -> int:
        """Get total number of turns."""
        return len(self.turns)

    def get_formatted_transcript(self) -> str:
        """Get human-readable formatted transcript."""
        lines = [
            f"Conversation ID: {self.conversation_id}",
            f"Patient ID: {self.patient_id}",
            f"Doctor Model: {self.doctor_model}",
            f"Ground Truth: {self.ground_truth_diagnosis}",
            f"Final Diagnosis: {self.final_diagnosis}",
            "-" * 50,
        ]

        for turn in self.turns:
            role_label = "Doctor" if turn.role == "doctor" else "Patient"
            lines.append(f"[Turn {turn.turn_number}] {role_label}: {turn.content}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "patient_id": self.patient_id,
            "doctor_model": self.doctor_model,
            "patient_data": self.patient_data,
            "turns": [t.to_dict() for t in self.turns],
            "final_diagnosis": self.final_diagnosis,
            "ground_truth_diagnosis": self.ground_truth_diagnosis,
            "noise_profile": self.noise_profile,
            "metadata": self.metadata,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transcript":
        """Create from dictionary."""
        turns_data = data.pop("turns", [])
        transcript = cls(**data)
        transcript.turns = [Turn.from_dict(t) for t in turns_data]
        return transcript

    def save(self, filepath: str) -> None:
        """Save transcript to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Transcript":
        """Load transcript from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def create_transcript(
        patient_data: Dict[str, Any],
        doctor_model: str,
        conversation_id: Optional[str] = None,
) -> Transcript:
    """
    Create a new transcript for a conversation.

    Args:
        patient_data: Patient case data
        doctor_model: Name of the doctor model
        conversation_id: Optional custom ID

    Returns:
        New Transcript instance
    """
    patient_id = patient_data.get("patient_id", "unknown")

    if conversation_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conversation_id = f"{patient_id}_{doctor_model}_{timestamp}"

    return Transcript(
        conversation_id=conversation_id,
        patient_id=patient_id,
        doctor_model=doctor_model,
        patient_data=patient_data,
        ground_truth_diagnosis=patient_data.get("diagnosis"),
        noise_profile=patient_data.get("noise_profile", []),
    )

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IntentName(StrEnum):
    GREETING = "greeting"
    SMALL_TALK = "small_talk"
    GENERAL_CONVERSATION = "general_conversation"
    CAPABILITIES = "capabilities"
    GRATITUDE = "gratitude"
    PRACTICE_INFORMATION = "practice_information"
    PROVIDER_LOOKUP = "provider_lookup"
    UNSUPPORTED_LOCAL_SEARCH = "unsupported_local_search"
    APPOINTMENT_REQUEST = "appointment_request"
    APPOINTMENT_CHANGE = "appointment_change"
    APPOINTMENT_CANCELLATION = "appointment_cancellation"
    CONFIRMATION = "confirmation"
    CORRECTION = "correction"
    HUMAN_HANDOFF = "human_handoff"
    HEALTHCARE_REQUEST = "healthcare_request"
    UNSUPPORTED_CLINICAL = "unsupported_clinical"
    URGENT_SAFETY = "urgent_safety"
    UNKNOWN = "unknown"


class DialogueAct(StrEnum):
    INFORM = "inform"
    REQUEST_INFORMATION = "request_information"
    CONFIRM = "confirm"
    CORRECT = "correct"
    CANCEL = "cancel"
    ACKNOWLEDGE = "acknowledge"
    ESCALATE = "escalate"


class PolicyDecision(StrEnum):
    ALLOW = "allow"
    REFUSE = "refuse"
    ESCALATE = "escalate"
    REQUIRE_CONFIRMATION = "require_confirmation"


class WorkflowName(StrEnum):
    RECEPTIONIST = "receptionist"
    SMALL_TALK = "small_talk"
    SAFETY_ESCALATION = "safety_escalation"
    GENERAL = "general"


class ReceptionistState(StrEnum):
    IDLE = "idle"
    COLLECTING_DETAILS = "collecting_details"
    REVIEWING_REQUEST = "reviewing_request"
    CONFIRMED = "confirmed"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    CORRECTION_REQUIRED = "correction_required"
    SUBMISSION_FAILED = "submission_failed"


class IntentClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: IntentName
    confidence: float = Field(ge=0, le=1)
    topic: str | None = None


class ExtractedSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    confidence: float = Field(ge=0, le=1)
    source: str = "conversation"


class PolicyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: PolicyDecision
    reason: str
    requires_confirmation: bool = False


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = False


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: str
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class WorkflowContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: WorkflowName
    state: ReceptionistState = ReceptionistState.IDLE
    next_action: str | None = None
    missing_fields: list[str] = Field(default_factory=list)


class ConversationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    workflow: WorkflowContext = Field(default_factory=lambda: WorkflowContext(name=WorkflowName.GENERAL))
    slots: dict[str, ExtractedSlot] = Field(default_factory=dict)
    provider_options: list[dict[str, Any]] = Field(default_factory=list)
    turn_count: int = Field(default=0, ge=0)


class OrchestratedTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: IntentClassification
    dialogue_act: DialogueAct
    entities: dict[str, ExtractedSlot] = Field(default_factory=dict)
    policy: PolicyResult
    workflow: WorkflowContext
    tool_call: ToolRequest | None = None
    tool_result: ToolResult | None = None
    response: str = Field(min_length=1)


class WorkflowDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: ConversationState
    transition: str
    next_action: str
    missing_fields: list[str] = Field(default_factory=list)
    tool_call: ToolRequest | None = None
    tool_result: ToolResult | None = None
    response: str = Field(min_length=1)

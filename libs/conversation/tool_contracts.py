"""Typed contracts for asynchronous conversation tools."""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from libs.conversation.domain import OperationName, OperationStatus


class ToolContractError(ValueError):
    """Raised when a tool request or result violates its contract."""


class ToolResultStatus(StrEnum):
    SUCCEEDED = OperationStatus.SUCCEEDED
    FAILED = OperationStatus.FAILED
    TIMED_OUT = OperationStatus.TIMED_OUT
    CANCELLED = OperationStatus.CANCELLED
    SUPERSEDED = OperationStatus.SUPERSEDED


class ToolErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    NOT_FOUND = "not_found"
    UPSTREAM_REJECTED = "upstream_rejected"
    AUTHORIZATION_FAILED = "authorization_failed"
    INTERNAL = "internal"


class ProviderRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    address: str | None = None
    category: str = Field(min_length=1)
    distance_km: float | None = Field(default=None, ge=0)
    phone: str | None = None
    website: str | None = None
    source: str = Field(min_length=1)
    source_record_id: str | None = None
    observed_at: datetime


class SearchProvidersInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: Literal[OperationName.SEARCH_PROVIDERS] = OperationName.SEARCH_PROVIDERS
    care_setting: str = Field(min_length=1)
    location: str = Field(min_length=1)
    appointment_reason: str | None = None


class CreateAppointmentRequestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: Literal[OperationName.CREATE_APPOINTMENT_REQUEST] = OperationName.CREATE_APPOINTMENT_REQUEST
    provider_id: str = Field(min_length=1)
    care_setting: str = Field(min_length=1)
    location: str = Field(min_length=1)
    appointment_reason: str = Field(min_length=1)
    preferred_time: str = Field(min_length=1)
    caller_name: str = Field(min_length=1)
    callback_number: str = Field(min_length=1)


class GetPracticeProfileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: Literal[OperationName.GET_PRACTICE_PROFILE] = OperationName.GET_PRACTICE_PROFILE


ToolInput = Annotated[
    Union[SearchProvidersInput, CreateAppointmentRequestInput, GetPracticeProfileInput],
    Field(discriminator="tool_name"),
]


class ToolError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ToolErrorCode
    message: str = Field(min_length=1)
    retryable: bool
    upstream: str | None = None
    retry_after_ms: int | None = Field(default=None, ge=0)


class ProviderSearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: Literal[OperationName.SEARCH_PROVIDERS] = OperationName.SEARCH_PROVIDERS
    resolved_location: str = Field(min_length=1)
    providers: list[ProviderRecord] = Field(default_factory=list)


class AppointmentRequestOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: Literal[OperationName.CREATE_APPOINTMENT_REQUEST] = OperationName.CREATE_APPOINTMENT_REQUEST
    request_reference: str = Field(min_length=1)
    status: Literal["submitted", "queued"]


class PracticeHoursOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    day: str = Field(min_length=1)
    opens_at: str | None = None
    closes_at: str | None = None
    closed: bool = False


class PracticeProfileOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: Literal[OperationName.GET_PRACTICE_PROFILE] = OperationName.GET_PRACTICE_PROFILE
    profile_id: str = Field(min_length=1)
    profile_version: int = Field(ge=1)
    display_name: str = Field(min_length=1)
    address: str | None = None
    phone: str | None = None
    website: str | None = None
    hours: list[PracticeHoursOutput] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    accepted_insurance: list[str] = Field(default_factory=list)


ToolOutput = Annotated[
    Union[ProviderSearchOutput, AppointmentRequestOutput, PracticeProfileOutput],
    Field(discriminator="tool_name"),
]


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: int = Field(default=1, ge=1)
    request_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    operation: OperationName
    requested_state_version: int = Field(ge=0)
    idempotency_key: str = Field(min_length=1)
    correlation_id: str = Field(min_length=1)
    input: ToolInput
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def input_matches_operation(self) -> "ToolRequest":
        if self.input.tool_name != self.operation:
            raise ToolContractError("tool input does not match operation")
        return self


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: int = Field(default=1, ge=1)
    request_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    operation: OperationName
    requested_state_version: int = Field(ge=0)
    correlation_id: str = Field(min_length=1)
    status: ToolResultStatus
    output: ToolOutput | None = None
    error: ToolError | None = None
    started_at: datetime | None = None
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def semantic_contract_is_valid(self) -> "ToolResult":
        return self.validate_semantics()

    def validate_semantics(self) -> "ToolResult":
        if self.status == ToolResultStatus.SUCCEEDED and self.output is None:
            raise ToolContractError("successful tool result requires typed output")
        if self.status != ToolResultStatus.SUCCEEDED and self.error is None:
            raise ToolContractError("non-successful tool result requires structured error")
        if self.output is not None and self.output.tool_name != self.operation:
            raise ToolContractError("tool output does not match operation")
        if self.error and self.status == ToolResultStatus.SUCCEEDED:
            raise ToolContractError("successful tool result cannot contain an error")
        return self

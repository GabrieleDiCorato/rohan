from pydantic import BaseModel


class ValidationResult(BaseModel):
    """Result of strategy validation."""

    is_valid: bool
    errors: list[str]

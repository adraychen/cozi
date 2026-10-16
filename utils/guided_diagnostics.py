"""Analysis-independent diagnostic records and deterministic overview status."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Diagnostic:
    diagnostic_id: str
    category: str
    severity: str
    title: str
    observed_result: str
    explanation: str
    recommended_action: str
    action_type: str | None = None
    action_target: str | None = None
    model_context: dict = field(default_factory=dict)


def diagnostic_status(items):
    if any(item.severity == 'blocking' for item in items):
        return 'Model issue'
    if any(item.severity in ('review', 'unresolved') for item in items):
        return 'Review advised'
    return 'Checks passed'

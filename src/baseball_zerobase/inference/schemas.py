from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PitchCandidate:
    pitch_type: str
    relative_zone: str

    @property
    def action(self) -> str:
        return f"{self.pitch_type}:{self.relative_zone}"


@dataclass(frozen=True, slots=True)
class PitchRecommendation:
    rank: int
    pitch_type: str
    relative_zone: str
    action: str
    ranking_score: float
    value_type: str
    explanation: dict[str, object]
    transition_distribution: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "pitch_type": self.pitch_type,
            "relative_zone": self.relative_zone,
            "action": self.action,
            "ranking_score": self.ranking_score,
            "value_type": self.value_type,
            "explanation": self.explanation,
            "transition_distribution": list(self.transition_distribution),
        }


@dataclass(frozen=True, slots=True)
class RecommendationReport:
    value_type: str
    zone_filtering: str
    candidate_count: int
    pitch_types: tuple[str, ...]
    model_training_manifest_hash: str | None
    recommendations: tuple[PitchRecommendation, ...]
    input_summary: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "value_type": self.value_type,
            "zone_filtering": self.zone_filtering,
            "candidate_count": self.candidate_count,
            "pitch_types": list(self.pitch_types),
            "model_training_manifest_hash": self.model_training_manifest_hash,
            "input_summary": self.input_summary,
            "recommendations": [
                recommendation.to_dict() for recommendation in self.recommendations
            ],
        }

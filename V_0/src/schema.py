from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# ==================== PYDANTIC SCHEMAS ====================

class ModeratorOpeningResponse(BaseModel):
    opening_statement: str = Field(
        min_length=50,
        description="Opening statement welcoming everyone and introducing the debate topic"
    )
    rules: List[str] = Field(
        min_length=3,
        max_length=5,
        description="List of exactly 3-5 debate rules as separate strings"
    )
    format_description: str = Field(
        min_length=30,
        description="Brief description of how the debate will proceed"
    )


class DebaterResponse(BaseModel):
    argument: str = Field(
        min_length=50,
        description="Main argument as a complete sentence or paragraph"
    )
    supporting_evidence: List[str] = Field(
        min_length=2,
        max_length=4,
        description="List of 2-4 supporting evidence points as separate strings"
    )
    counterpoints: List[str] = Field(
        default_factory=list,
        description="Optional list of counterpoints to opponent"
    )
    conclusion: str = Field(
        default="",
        description="Optional brief conclusion"
    )


class ModeratorRoundResponse(BaseModel):
    round_number: int = Field(description="Current round number")
    debater_evaluated: str = Field(description="Which debater is being evaluated")
    summary: str = Field(
        min_length=30,
        description="Summary of the argument presented"
    )
    score: int = Field(
        ge=0,
        le=10,
        description="Score from 0 to 10"
    )
    validation_notes: str = Field(
        default="No issues found",
        description="Notes on factual accuracy"
    )
    key_points: List[str] = Field(
        min_length=2,
        max_length=3,
        description="List of 2-3 key points made"
    )


class ModeratorFinalResponse(BaseModel):
    final_summary: str = Field(
        min_length=100,
        description="Comprehensive summary of the entire debate"
    )
    winner: Literal["Debater 1", "Debater 2", "Tie"] = Field(
        description="The winner - must be exactly 'Debater 1', 'Debater 2', or 'Tie'"
    )
    reasoning: str = Field(
        min_length=50,
        description="Explanation for the decision"
    )
    debater1_total_score: int = Field(description="Debater 1 total score")
    debater2_total_score: int = Field(description="Debater 2 total score")
    key_moments: List[str] = Field(
        min_length=2,
        max_length=3,
        description="List of 2-3 key moments that influenced the outcome"
    )

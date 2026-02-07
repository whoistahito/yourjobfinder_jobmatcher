from typing import List, Optional

from pydantic import BaseModel, Field


class SkillExperienceBase(BaseModel):
    """Base class for skill and experience data structures."""
    skills: List[str] = Field(
        default_factory=list,
        description="List of required skills"
    )
    experiences: List[str] = Field(
        default_factory=list,
        description="List of experience requirements"
    )
    qualifications: List[str] = Field(
        default_factory=list,
        description="List of qualifications/certifications"
    )


class Requirements(SkillExperienceBase):
    """Normalized requirements schema for job extraction outputs."""
    pass


class UserProfile(SkillExperienceBase):
    """User profile with skills, experiences, and qualifications."""
    pass


class ExtractionPipeline(BaseModel):
    extractorModelIds: List[str] = Field(..., min_length=2, description="Two model ids used for extraction")
    judgeModelId: str = Field(..., description="Model id used to judge/validate merged requirements")


class JobExtractionInput(BaseModel):
    inputText: str = Field(...,
                           description="Text to extract job requirements from"
                           )
    modelId: str = Field(...,
                         description="LLM model id to use"
                         )
    extractionPipeline: Optional[ExtractionPipeline] = Field(
        default=None,
        description="Optional pipeline override: run two extractors, merge, then judge",
    )
    userProfile: UserProfile = Field(...,
                                     description="User profile data"
                                     )


class SimilarityScore(BaseModel):
    score: float = Field(...,
                         description="Similarity score")


class JobMatchingResponse(BaseModel):
    jobRequirements: Requirements = Field(...,
                                          description="Job requirements")
    userProfile: UserProfile = Field(...,
                                     description="User profile data")
    similarityScore: SimilarityScore = Field(...,
                                             description="Similarity score")

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


class GeneratedAnswer(BaseModel):
    answer: str
    insufficient_evidence: bool
    citations: list[str]


class Citation(BaseModel):
    chunk_id: str
    page_number: int
    section: str
    text: str
    bboxes: list[list[float]]


class AskResponse(BaseModel):
    document_id: str
    question: str
    answer: str
    insufficient_evidence: bool
    citations: list[Citation]
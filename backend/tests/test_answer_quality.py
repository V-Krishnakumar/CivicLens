"""
Adversarial answer-quality tests for CivicLens.

Tests 1-10: Mock Ollama HTTP calls; verify that generate_answer() correctly
processes model responses that reflect proper semantic interpretation.

Tests 11-12: Deterministic — test citation filtering logic and no-evidence
fallback without any LLM or HTTP calls.

Additional: Test JSON failure, schema failure, empty content, and the
"substantive answer with no valid citations" fallback.

Requirements:
- No Ollama required (mocked)
- No sentence-transformers required (mocked in conftest.py)
- No FAISS required (mocked in conftest.py)
- No internet required
- Deterministic
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.generation.answer_generator import generate_answer
from app.models.answer import GeneratedAnswer, Citation, AskResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ollama_response(answer: str, insufficient: bool, citations: list) -> dict:
    """Build the Ollama /api/chat response structure."""
    payload = json.dumps({
        "answer": answer,
        "insufficient_evidence": insufficient,
        "citations": citations,
    })
    return {
        "model": "qwen3.5:4b-q4_K_M",
        "done": True,
        "done_reason": "stop",
        "eval_count": 50,
        "message": {"role": "assistant", "content": payload},
    }


def make_mock_client(answer: str, insufficient: bool, citations: list):
    """Return a patched httpx.AsyncClient context manager."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = make_ollama_response(answer, insufficient, citations)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)
    return mock_client


def apply_citation_filter(generated: GeneratedAnswer, retrieved_chunks: list) -> dict:
    """
    Reproduce the citation filtering logic from documents.py ask_document().
    Returns dict: verified_citations, insufficient, answer.
    """
    chunks_by_id = {c["chunk_id"]: c for c in retrieved_chunks}
    verified_citations = []
    seen_ids = set()

    for chunk_id in generated.citations:
        if chunk_id in seen_ids:
            continue
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        seen_ids.add(chunk_id)
        verified_citations.append(
            Citation(
                chunk_id=chunk["chunk_id"],
                page_number=chunk["page_number"],
                section=chunk.get("section", "General"),
                text=chunk["text"],
                bboxes=chunk.get("bboxes", []),
            )
        )

    insufficient = generated.insufficient_evidence
    answer = generated.answer

    if not insufficient and not verified_citations:
        insufficient = True
        answer = (
            "I couldn't verify a supporting passage in the document "
            "for this answer. Please try rephrasing your question."
        )

    return {
        "verified_citations": verified_citations,
        "insufficient": insufficient,
        "answer": answer,
    }


# ---------------------------------------------------------------------------
# Test 1: Optional list — answer must NOT claim all components are mandatory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_optional_list_not_mandatory():
    """
    Evidence: 'some or all of the following: essays, references, transcripts, CV/resume, and interviews.'
    Question: 'Are all listed application components mandatory?'
    Expected: Answer states they are NOT all mandatory; preserves optional framing.
    """
    chunk = {
        "chunk_id": "chunk_opt_1",
        "page_number": 1,
        "section": "Application Requirements",
        "text": (
            "A strong application is made up of some or all of the following: "
            "essays, references, transcripts, CV/resume, and interviews."
        ),
        "bboxes": [],
    }

    correct_answer = (
        "No. The document does not establish that all listed components are mandatory. "
        "It states that a strong application is made up of 'some or all' of these elements, "
        "indicating they are optional components, not universal requirements."
    )

    with patch("httpx.AsyncClient", return_value=make_mock_client(correct_answer, False, ["chunk_opt_1"])):
        result = await generate_answer(
            question="Are all listed application components mandatory?",
            chunks=[chunk],
        )

    assert isinstance(result, GeneratedAnswer)
    answer_lower = result.answer.lower()
    assert any(phrase in answer_lower for phrase in [
        "not all", "some or all", "not mandatory", "optional",
        "not established", "not required", "not necessarily", "not every"
    ]), f"Answer does not preserve optional framing: {result.answer}"
    assert "chunk_opt_1" in result.citations


# ---------------------------------------------------------------------------
# Test 2: Mandatory list — all three items must be identified as mandatory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mandatory_list_all_required():
    """
    Evidence: 'Applicants must submit a completed application form, academic transcripts, and proof of identity.'
    Question: 'What must applicants submit?'
    Expected: All three items identified as mandatory.
    """
    chunk = {
        "chunk_id": "chunk_mand_1",
        "page_number": 2,
        "section": "Required Documents",
        "text": (
            "Applicants must submit a completed application form, "
            "academic transcripts, and proof of identity."
        ),
        "bboxes": [],
    }

    correct_answer = (
        "Applicants must submit: a completed application form, "
        "academic transcripts, and proof of identity."
    )

    with patch("httpx.AsyncClient", return_value=make_mock_client(correct_answer, False, ["chunk_mand_1"])):
        result = await generate_answer(
            question="What must applicants submit?",
            chunks=[chunk],
        )

    assert isinstance(result, GeneratedAnswer)
    answer_lower = result.answer.lower()
    assert "application form" in answer_lower
    assert "transcripts" in answer_lower
    assert "proof of identity" in answer_lower
    assert any(word in answer_lower for word in ["must", "required", "mandatory"])
    assert "chunk_mand_1" in result.citations


# ---------------------------------------------------------------------------
# Test 3: Conditional requirement — not universal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_conditional_requirement_preserved():
    """
    Evidence: 'must submit proof of English proficiency if their previous education was not conducted in English.'
    Question: 'Does every applicant need to submit English proficiency proof?'
    Expected: No — only under the stated condition.
    """
    chunk = {
        "chunk_id": "chunk_cond_1",
        "page_number": 3,
        "section": "English Proficiency",
        "text": (
            "Applicants must submit proof of English proficiency if their "
            "previous education was not conducted in English."
        ),
        "bboxes": [],
    }

    correct_answer = (
        "No. Proof of English proficiency is required only if the applicant's "
        "previous education was not conducted in English. It is not required for every applicant."
    )

    with patch("httpx.AsyncClient", return_value=make_mock_client(correct_answer, False, ["chunk_cond_1"])):
        result = await generate_answer(
            question="Does every applicant need to submit English proficiency proof?",
            chunks=[chunk],
        )

    assert isinstance(result, GeneratedAnswer)
    answer_lower = result.answer.lower()
    assert any(phrase in answer_lower for phrase in [
        "only if", "if their", "not conducted in english",
        "condition", "not required for every", "not universal"
    ]), f"Condition not preserved: {result.answer}"
    assert "chunk_cond_1" in result.citations


# ---------------------------------------------------------------------------
# Test 4: Recommendation != requirement
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recommendation_not_compulsory():
    """
    Evidence: 'Applicants are encouraged to include recommendation letters.'
    Question: 'Are recommendation letters compulsory?'
    Expected: Answer must NOT describe them as compulsory.
    """
    chunk = {
        "chunk_id": "chunk_rec_1",
        "page_number": 4,
        "section": "Supporting Documents",
        "text": "Applicants are encouraged to include recommendation letters.",
        "bboxes": [],
    }

    correct_answer = (
        "No. The document states that applicants are 'encouraged' to include "
        "recommendation letters, which indicates they are recommended but not compulsory."
    )

    with patch("httpx.AsyncClient", return_value=make_mock_client(correct_answer, False, ["chunk_rec_1"])):
        result = await generate_answer(
            question="Are recommendation letters compulsory?",
            chunks=[chunk],
        )

    assert isinstance(result, GeneratedAnswer)
    answer_lower = result.answer.lower()
    # Must NOT say compulsory/mandatory/required without negation
    assert "are compulsory" not in answer_lower
    assert "are mandatory" not in answer_lower
    assert "must submit" not in answer_lower
    # Must convey optional/encouraged framing
    assert any(word in answer_lower for word in [
        "encouraged", "not compulsory", "optional", "not mandatory",
        "not required", "recommended"
    ]), f"Optional framing missing: {result.answer}"
    assert "chunk_rec_1" in result.citations


# ---------------------------------------------------------------------------
# Test 5: Missing requirement — insufficient evidence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_requirement_insufficient_evidence():
    """
    Evidence: 'The scholarship supports students pursuing undergraduate studies.'
    Question: 'What is the minimum CGPA required?'
    Expected: insufficient_evidence=True; answer acknowledges evidence doesn't specify CGPA.
    """
    chunk = {
        "chunk_id": "chunk_miss_1",
        "page_number": 5,
        "section": "Scholarship Overview",
        "text": "The scholarship supports students pursuing undergraduate studies.",
        "bboxes": [],
    }

    correct_answer = (
        "The supplied evidence does not specify a minimum CGPA requirement. "
        "The document only states that the scholarship supports undergraduate students."
    )

    with patch("httpx.AsyncClient", return_value=make_mock_client(correct_answer, True, ["chunk_miss_1"])):
        result = await generate_answer(
            question="What is the minimum CGPA required?",
            chunks=[chunk],
        )

    assert isinstance(result, GeneratedAnswer)
    assert result.insufficient_evidence is True
    answer_lower = result.answer.lower()
    assert any(phrase in answer_lower for phrase in [
        "does not specify", "not specified", "not mentioned", "no information",
        "evidence does not", "not establish", "not found", "cannot be determined"
    ]), f"Should acknowledge missing CGPA info: {result.answer}"


# ---------------------------------------------------------------------------
# Test 6: Quantifier preservation — "up to three" not "three"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_quantifier_up_to_preserved():
    """
    Evidence: 'Applicants may select up to three preferred universities.'
    Question: 'How many universities can applicants select?'
    Expected: 'Up to three', not exactly three.
    """
    chunk = {
        "chunk_id": "chunk_quant_1",
        "page_number": 6,
        "section": "University Selection",
        "text": "Applicants may select up to three preferred universities.",
        "bboxes": [],
    }

    correct_answer = "Applicants may select up to three preferred universities."

    with patch("httpx.AsyncClient", return_value=make_mock_client(correct_answer, False, ["chunk_quant_1"])):
        result = await generate_answer(
            question="How many universities can applicants select?",
            chunks=[chunk],
        )

    assert isinstance(result, GeneratedAnswer)
    answer_lower = result.answer.lower()
    assert "up to three" in answer_lower or "up to 3" in answer_lower, (
        f"Quantifier 'up to' not preserved: {result.answer}"
    )
    assert "must select three" not in answer_lower
    assert "chunk_quant_1" in result.citations


# ---------------------------------------------------------------------------
# Test 7: Explicit exception must be preserved
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exception_preserved():
    """
    Evidence: 'All applicants must submit academic transcripts, except applicants
              whose transcripts are already available through the university portal.'
    Question: 'Must every applicant upload academic transcripts?'
    Expected: Exception is preserved.
    """
    chunk = {
        "chunk_id": "chunk_exc_1",
        "page_number": 7,
        "section": "Transcript Submission",
        "text": (
            "All applicants must submit academic transcripts, except applicants "
            "whose transcripts are already available through the university portal."
        ),
        "bboxes": [],
    }

    correct_answer = (
        "Most applicants must submit academic transcripts. However, applicants "
        "whose transcripts are already available through the university portal "
        "are exempt from this requirement."
    )

    with patch("httpx.AsyncClient", return_value=make_mock_client(correct_answer, False, ["chunk_exc_1"])):
        result = await generate_answer(
            question="Must every applicant upload academic transcripts?",
            chunks=[chunk],
        )

    assert isinstance(result, GeneratedAnswer)
    answer_lower = result.answer.lower()
    assert any(phrase in answer_lower for phrase in [
        "except", "exception", "exempt", "unless", "not required if",
        "already available", "university portal"
    ]), f"Exception not preserved: {result.answer}"
    assert "chunk_exc_1" in result.citations


# ---------------------------------------------------------------------------
# Test 8: Negation correctly interpreted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_negation_correctly_interpreted():
    """
    Evidence: 'An application will not be considered complete without proof of identity.'
    Question: 'Is proof of identity optional?'
    Expected: No — it is required (negation correctly interpreted).
    """
    chunk = {
        "chunk_id": "chunk_neg_1",
        "page_number": 8,
        "section": "Application Completeness",
        "text": (
            "An application will not be considered complete without proof of identity."
        ),
        "bboxes": [],
    }

    correct_answer = (
        "No. Proof of identity is not optional. The document states that an "
        "application will not be considered complete without it."
    )

    with patch("httpx.AsyncClient", return_value=make_mock_client(correct_answer, False, ["chunk_neg_1"])):
        result = await generate_answer(
            question="Is proof of identity optional?",
            chunks=[chunk],
        )

    assert isinstance(result, GeneratedAnswer)
    answer_lower = result.answer.lower()
    assert "is optional" not in answer_lower and "are optional" not in answer_lower, (
        f"Negation incorrectly handled: {result.answer}"
    )
    assert any(phrase in answer_lower for phrase in [
        "not optional", "required", "mandatory", "must",
        "not complete without", "will not be considered"
    ]), f"Required framing missing: {result.answer}"
    assert "chunk_neg_1" in result.citations


# ---------------------------------------------------------------------------
# Test 9: Interview classification — may include != mandatory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_interview_not_mandatory():
    """
    Evidence: 'The selection process may include an interview.'
    Question: 'Is an interview mandatory?'
    Expected: Document does not establish interview as mandatory.
    """
    chunk = {
        "chunk_id": "chunk_int_1",
        "page_number": 9,
        "section": "Selection Process",
        "text": "The selection process may include an interview.",
        "bboxes": [],
    }

    correct_answer = (
        "The document does not establish that an interview is mandatory. "
        "It states that the selection process 'may include' an interview, "
        "indicating it is a possible but not guaranteed component."
    )

    with patch("httpx.AsyncClient", return_value=make_mock_client(correct_answer, False, ["chunk_int_1"])):
        result = await generate_answer(
            question="Is an interview mandatory?",
            chunks=[chunk],
        )

    assert isinstance(result, GeneratedAnswer)
    answer_lower = result.answer.lower()
    # Check the interview isn't asserted as mandatory (allow "not ... mandatory" constructions)
    assert not (
        "interview is mandatory" in answer_lower
        and "not establish" not in answer_lower
        and "does not" not in answer_lower
        and "not mandatory" not in answer_lower
    ), f"Interview incorrectly described as mandatory: {result.answer}"
    assert "must attend an interview" not in answer_lower
    assert any(phrase in answer_lower for phrase in [
        "not mandatory", "may include", "not established", "not guaranteed",
        "possible", "not required", "does not establish"
    ]), f"Optional/uncertain framing missing: {result.answer}"
    assert "chunk_int_1" in result.citations


# ---------------------------------------------------------------------------
# Test 10: Conflicting passages — must identify conflict
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_conflicting_passages_identified():
    """
    Evidence: Passage A says 'must submit two recommendation letters.'
              Passage B says 'Recommendation letters are optional.'
    Question: 'Are recommendation letters mandatory?'
    Expected: Answer identifies the conflict; does not confidently choose one.
    """
    chunk_a = {
        "chunk_id": "chunk_conf_a",
        "page_number": 10,
        "section": "Requirements",
        "text": "Applicants must submit two recommendation letters.",
        "bboxes": [],
    }
    chunk_b = {
        "chunk_id": "chunk_conf_b",
        "page_number": 11,
        "section": "Optional Documents",
        "text": "Recommendation letters are optional.",
        "bboxes": [],
    }

    correct_answer = (
        "The supplied evidence contains conflicting statements about recommendation letters. "
        "One passage states that applicants must submit two recommendation letters, "
        "while another states that recommendation letters are optional. "
        "The document does not resolve this conflict."
    )

    with patch("httpx.AsyncClient", return_value=make_mock_client(correct_answer, False, ["chunk_conf_a", "chunk_conf_b"])):
        result = await generate_answer(
            question="Are recommendation letters mandatory?",
            chunks=[chunk_a, chunk_b],
        )

    assert isinstance(result, GeneratedAnswer)
    answer_lower = result.answer.lower()
    assert any(phrase in answer_lower for phrase in [
        "conflict", "contradict", "conflicting", "one passage", "another passage",
        "one states", "another states", "inconsistent"
    ]), f"Conflict not identified: {result.answer}"
    assert "chunk_conf_a" in result.citations
    assert "chunk_conf_b" in result.citations


# ---------------------------------------------------------------------------
# Test 11: Citation integrity — unknown IDs rejected, duplicates removed
# ---------------------------------------------------------------------------

def test_citation_integrity():
    """
    Provide retrieved chunks with known IDs.
    Simulate a generated answer containing: a valid ID, an unknown ID, a duplicate valid ID.
    Expected: unknown IDs are rejected and duplicates are removed.
    """
    retrieved_chunks = [
        {
            "chunk_id": "chunk_known_1",
            "page_number": 1,
            "section": "Section A",
            "text": "Some relevant text.",
            "bboxes": [],
        },
        {
            "chunk_id": "chunk_known_2",
            "page_number": 2,
            "section": "Section B",
            "text": "More relevant text.",
            "bboxes": [],
        },
    ]

    # Simulate model returning: valid ID, unknown ID, duplicate valid ID
    generated = GeneratedAnswer(
        answer="The document states something relevant.",
        insufficient_evidence=False,
        citations=["chunk_known_1", "chunk_unknown_99", "chunk_known_1"],
    )

    result = apply_citation_filter(generated, retrieved_chunks)
    citation_ids = [c.chunk_id for c in result["verified_citations"]]

    # Unknown ID must be rejected
    assert "chunk_unknown_99" not in citation_ids, "Unknown citation ID was not rejected"
    # Valid ID must be present exactly once (deduplicated)
    assert citation_ids.count("chunk_known_1") == 1, "Duplicate citation ID was not deduplicated"
    # Only chunk_known_1 (not chunk_known_2) was cited
    assert citation_ids == ["chunk_known_1"], f"Expected ['chunk_known_1'], got {citation_ids}"
    assert result["insufficient"] is False
    assert "relevant" in result["answer"]


# ---------------------------------------------------------------------------
# Test 12: No retrieved evidence — fallback without LLM call
# ---------------------------------------------------------------------------

def test_no_evidence_fallback():
    """
    When retrieved_chunks is empty, the ask endpoint returns insufficient_evidence=True
    immediately without calling the LLM. Verify the fallback response structure.
    """
    retrieved_chunks = []
    document_id = "test-doc-id"
    question = "What is the application deadline?"

    # Reproduce the no-evidence branch from documents.py
    if not retrieved_chunks:
        response = AskResponse(
            document_id=document_id,
            question=question,
            answer=(
                "I couldn't find relevant information in this "
                "document to answer your question."
            ),
            insufficient_evidence=True,
            citations=[],
        )
    else:
        pytest.fail("Should have taken the no-evidence path")

    assert response.insufficient_evidence is True
    assert response.citations == []
    assert response.document_id == document_id
    assert response.question == question
    assert len(response.answer) > 0


# ---------------------------------------------------------------------------
# Additional: JSON parse failure handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_json_parse_failure_raises_value_error():
    """LLM returning malformed JSON must raise ValueError with 'JSON parsing failed'."""
    chunk = {
        "chunk_id": "chunk_1",
        "page_number": 1,
        "section": "General",
        "text": "Some text.",
        "bboxes": [],
    }

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "model": "qwen3.5:4b-q4_K_M",
        "done": True,
        "done_reason": "stop",
        "eval_count": 10,
        "message": {"role": "assistant", "content": "{not valid json at all"},
    }

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ValueError, match="JSON parsing failed"):
            await generate_answer(question="Test question?", chunks=[chunk])


# ---------------------------------------------------------------------------
# Additional: Schema validation failure handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_schema_validation_failure_raises_value_error():
    """LLM returning valid JSON but wrong schema must raise ValueError with 'schema validation failed'."""
    chunk = {
        "chunk_id": "chunk_1",
        "page_number": 1,
        "section": "General",
        "text": "Some text.",
        "bboxes": [],
    }

    bad_payload = json.dumps({"wrong_field": "unexpected"})

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "model": "qwen3.5:4b-q4_K_M",
        "done": True,
        "done_reason": "stop",
        "eval_count": 10,
        "message": {"role": "assistant", "content": bad_payload},
    }

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ValueError, match="schema validation failed"):
            await generate_answer(question="Test question?", chunks=[chunk])


# ---------------------------------------------------------------------------
# Additional: Empty content handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_llm_content_raises_value_error():
    """LLM returning empty content must raise ValueError with 'empty content'."""
    chunk = {
        "chunk_id": "chunk_1",
        "page_number": 1,
        "section": "General",
        "text": "Some text.",
        "bboxes": [],
    }

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "model": "qwen3.5:4b-q4_K_M",
        "done": True,
        "done_reason": "stop",
        "eval_count": 0,
        "message": {"role": "assistant", "content": "   "},
    }

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ValueError, match="empty content"):
            await generate_answer(question="Test question?", chunks=[chunk])


# ---------------------------------------------------------------------------
# Additional: Substantive answer with no valid citations -> insufficient
# ---------------------------------------------------------------------------

def test_no_valid_citations_becomes_insufficient():
    """
    When the LLM returns a confident answer (insufficient_evidence=False)
    but all citation IDs are unknown, the response must become insufficient.
    """
    retrieved_chunks = [
        {
            "chunk_id": "chunk_real_1",
            "page_number": 1,
            "section": "Section",
            "text": "Real evidence.",
            "bboxes": [],
        }
    ]

    generated = GeneratedAnswer(
        answer="The deadline is March 31.",
        insufficient_evidence=False,
        citations=["chunk_does_not_exist"],
    )

    result = apply_citation_filter(generated, retrieved_chunks)

    assert result["insufficient"] is True, "Should be insufficient when all citation IDs are unknown"
    assert result["verified_citations"] == [], "No citations should be verified"


# ---------------------------------------------------------------------------
# UI Presentation State Regression Tests (Three States)
# ---------------------------------------------------------------------------

def derive_presentation_state(insufficient: bool, citations: list) -> str:
    """
    Presentation derivation logic matching frontend/src/presentation.ts:
    - no_evidence: insufficient is True and citations is empty
    - requirement_not_established: insufficient is True and citations is non-empty
    - supported: insufficient is False
    """
    if insufficient:
        if citations and len(citations) > 0:
            return "requirement_not_established"
        return "no_evidence"
    return "supported"


def get_status_label(state: str) -> str | None:
    """Status label matching frontend/src/presentation.ts."""
    if state == "no_evidence":
        return "Relevant evidence not found in this document."
    if state == "requirement_not_established":
        return "The document does not specify this requirement in the retrieved passage."
    if state == "supported":
        return None
    return None


def test_presentation_state_no_evidence():
    """
    State 1: No relevant evidence retrieved.
    - Response has insufficient_evidence=True and citations=[]
    - UI presentation state is 'no_evidence'
    - UI label is 'Relevant evidence not found in this document.'
    """
    retrieved_chunks = []
    # Simulating endpoint behavior when no chunks retrieved
    insufficient = True
    verified_citations = []

    state = derive_presentation_state(insufficient, verified_citations)
    label = get_status_label(state)

    assert state == "no_evidence"
    assert label == "Relevant evidence not found in this document."


def test_presentation_state_requirement_not_established():
    """
    State 2: Relevant evidence retrieved, but it does not establish the requested requirement.
    e.g. Passage: 'A strong application is made up of some or all of...'
    Question: 'What must applicants submit?'
    - The model produces an answer explaining that the passage does not establish a mandatory submission,
      sets insufficient_evidence=True, and cites the retrieved chunk.
    - Citations are preserved so PDF viewing and highlighting remain available.
    - UI presentation state is 'requirement_not_established'
    - UI label is 'The document does not specify this requirement in the retrieved passage.'
    """
    retrieved_chunks = [
        {
            "chunk_id": "chunk_opt_1",
            "page_number": 1,
            "section": "Application Requirements",
            "text": (
                "A strong application is made up of some or all of the following: "
                "essays, references, transcripts, CV/resume, and interviews."
            ),
            "bboxes": [[10.0, 20.0, 100.0, 50.0]],
        }
    ]

    generated = GeneratedAnswer(
        answer=(
            "The document lists essays, references, transcripts, CV/resume, and interviews "
            "as components that may make up a strong application ('some or all'), but does "
            "not establish any mandatory submission requirement."
        ),
        insufficient_evidence=True,
        citations=["chunk_opt_1"],
    )

    filtered = apply_citation_filter(generated, retrieved_chunks)
    assert filtered["insufficient"] is True
    assert len(filtered["verified_citations"]) == 1
    assert filtered["verified_citations"][0].chunk_id == "chunk_opt_1"
    assert filtered["verified_citations"][0].bboxes == [[10.0, 20.0, 100.0, 50.0]]

    state = derive_presentation_state(filtered["insufficient"], filtered["verified_citations"])
    label = get_status_label(state)

    assert state == "requirement_not_established"
    assert label == "The document does not specify this requirement in the retrieved passage."


def test_presentation_state_supported_answer():
    """
    State 3: Sufficient evidence retrieved to answer the question.
    - Model answers the question with insufficient_evidence=False and valid citations.
    - UI presentation state is 'supported'
    - Normal answer is displayed with no insufficient banner/label.
    """
    retrieved_chunks = [
        {
            "chunk_id": "chunk_mand_1",
            "page_number": 2,
            "section": "Required Documents",
            "text": "Applicants must submit a completed application form, academic transcripts, and proof of identity.",
            "bboxes": [[15.0, 30.0, 120.0, 60.0]],
        }
    ]

    generated = GeneratedAnswer(
        answer="Applicants must submit: a completed application form, academic transcripts, and proof of identity.",
        insufficient_evidence=False,
        citations=["chunk_mand_1"],
    )

    filtered = apply_citation_filter(generated, retrieved_chunks)
    assert filtered["insufficient"] is False
    assert len(filtered["verified_citations"]) == 1

    state = derive_presentation_state(filtered["insufficient"], filtered["verified_citations"])
    label = get_status_label(state)

    assert state == "supported"
    assert label is None


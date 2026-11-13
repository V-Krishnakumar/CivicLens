"""
Tests for PDF chunking, context preservation, and boundary integrity.

Regression test cases:
1. A condition split across a potential chunk boundary.
2. A sentence containing "some or all".
3. An exception containing "unless" or "except".
4. A negation split across a boundary.
5. A multi-page paragraph.
6. Citation page numbers and bounding boxes.
7. Overlapping chunks and duplicate retrieval.
"""

import pytest
from app.ingestion.chunker import create_chunks, split_into_sentences
from app.retrieval.search import search_document


def test_split_into_sentences_basic():
    text = "First sentence. Second sentence! Third sentence? Fourth sentence."
    sentences = split_into_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "First sentence."
    assert sentences[1] == "Second sentence!"
    assert sentences[2] == "Third sentence?"
    assert sentences[3] == "Fourth sentence."


def test_condition_not_split_across_boundary():
    """
    Test 1: A condition split across a potential chunk boundary.
    e.g. 'Applicants must submit proof of English proficiency if their previous education was not conducted in English.'
    Verify that the sentence is preserved as a complete unit and not sliced mid-condition.
    """
    sentence = "Applicants must submit proof of English proficiency if their previous education was not conducted in English."
    # Small max_chars that would force a split if splitting arbitrarily
    pages = [
        {
            "page_number": 1,
            "blocks": [
                {
                    "text": sentence,
                    "bbox": [10.0, 20.0, 200.0, 50.0],
                    "font_size": 11.0,
                    "is_bold": False,
                }
            ],
        }
    ]

    chunks = create_chunks(pages, max_chars=120, overlap_chars=30)
    assert len(chunks) == 1
    assert "if their previous education was not conducted in English" in chunks[0]["text"]
    assert "Applicants must submit proof" in chunks[0]["text"]


def test_sentence_containing_some_or_all_preserved():
    """
    Test 2: A sentence containing 'some or all'.
    Verify that 'some or all' and its list elements stay intact together.
    """
    sentence1 = "Introduction to the program requirements and guidelines."
    sentence2 = "A strong application is made up of some or all of the following: essays, references, transcripts, CV/resume, and interviews."
    
    pages = [
        {
            "page_number": 1,
            "blocks": [
                {
                    "text": f"{sentence1} {sentence2}",
                    "bbox": [10.0, 20.0, 300.0, 80.0],
                    "font_size": 11.0,
                    "is_bold": False,
                }
            ],
        }
    ]

    chunks = create_chunks(pages, max_chars=150, overlap_chars=40)
    # Ensure that sentence2 is intact in at least one chunk without losing 'some or all'
    target_chunks = [c for c in chunks if "some or all" in c["text"]]
    assert len(target_chunks) >= 1
    for tc in target_chunks:
        assert "some or all of the following: essays, references, transcripts, CV/resume, and interviews." in tc["text"]


def test_exception_containing_unless_or_except_preserved():
    """
    Test 3: An exception containing 'unless' or 'except'.
    Verify the exception is not separated from the obligation it modifies.
    """
    sentence1 = "All applicants must submit academic transcripts, except applicants whose transcripts are already available through the university portal."
    sentence2 = "Late submissions will not be reviewed under any circumstances unless approved in advance by the dean."

    pages = [
        {
            "page_number": 1,
            "blocks": [
                {
                    "text": f"{sentence1}\n{sentence2}",
                    "bbox": [15.0, 25.0, 350.0, 100.0],
                    "font_size": 11.0,
                    "is_bold": False,
                }
            ],
        }
    ]

    chunks = create_chunks(pages, max_chars=160, overlap_chars=50)
    # Check that 'except' clause is within the transcript sentence chunk
    transcript_chunk = next(c for c in chunks if "academic transcripts" in c["text"])
    assert "except applicants whose transcripts are already available" in transcript_chunk["text"]

    # Check that 'unless' clause is within the late submissions sentence chunk
    late_chunk = next(c for c in chunks if "Late submissions" in c["text"])
    assert "unless approved in advance by the dean" in late_chunk["text"]


def test_negation_split_across_boundary():
    """
    Test 4: A negation split across a boundary.
    e.g. 'An application will not be considered complete without proof of identity.'
    Verify the negation ('will not be considered complete without') stays intact.
    """
    text = "General eligibility criteria apply to all candidates. An application will not be considered complete without proof of identity. Further instructions follow below."
    pages = [
        {
            "page_number": 2,
            "blocks": [
                {
                    "text": text,
                    "bbox": [20.0, 30.0, 400.0, 90.0],
                    "font_size": 10.0,
                    "is_bold": False,
                }
            ],
        }
    ]

    chunks = create_chunks(pages, max_chars=120, overlap_chars=30)
    negation_chunk = next(c for c in chunks if "without proof of identity" in c["text"])
    assert "will not be considered complete without proof of identity" in negation_chunk["text"]


def test_multi_page_paragraph_continuity():
    """
    Test 5: A multi-page paragraph.
    Verify that content flowing across page 1 and page 2 is chunked properly,
    attributing primary page number and keeping text continuous.
    """
    pages = [
        {
            "page_number": 1,
            "blocks": [
                {
                    "text": "The committee will evaluate every submission based on academic merit.",
                    "bbox": [50.0, 700.0, 500.0, 750.0],
                    "font_size": 11.0,
                    "is_bold": False,
                }
            ],
        },
        {
            "page_number": 2,
            "blocks": [
                {
                    "text": "Furthermore, candidates with extensive volunteer experience will be prioritized.",
                    "bbox": [50.0, 50.0, 500.0, 100.0],
                    "font_size": 11.0,
                    "is_bold": False,
                }
            ],
        },
    ]

    chunks = create_chunks(pages, max_chars=300, overlap_chars=50)
    assert len(chunks) == 1
    assert "academic merit." in chunks[0]["text"]
    assert "volunteer experience will be prioritized." in chunks[0]["text"]
    assert chunks[0]["page_number"] == 1
    assert len(chunks[0]["bboxes"]) == 2


def test_citation_page_numbers_and_bounding_boxes():
    """
    Test 6: Citation page numbers and bounding boxes.
    Verify that page numbers, sections, and deduplicated bboxes are properly populated.
    """
    pages = [
        {
            "page_number": 3,
            "blocks": [
                {
                    "text": "FINANCIAL ASSISTANCE",
                    "bbox": [50.0, 40.0, 250.0, 60.0],
                    "font_size": 16.0,
                    "is_bold": True,
                },
                {
                    "text": "Need-based grants are available to eligible domestic students. Applications open in September.",
                    "bbox": [50.0, 70.0, 450.0, 110.0],
                    "font_size": 11.0,
                    "is_bold": False,
                },
            ],
        }
    ]

    chunks = create_chunks(pages, max_chars=500, overlap_chars=100)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["page_number"] == 3
    assert chunk["section"] == "FINANCIAL ASSISTANCE"
    assert "Need-based grants are available" in chunk["text"]
    assert chunk["bboxes"] == [[50.0, 70.0, 450.0, 110.0]]


def test_overlapping_chunks_and_duplicate_retrieval(tmp_path, monkeypatch):
    """
    Test 7: Overlapping chunks and duplicate retrieval.
    Verify that when chunks overlap in content, the retrieval search_document
    deduplicates near-identical/redundant chunk results.
    """
    import json

    metadata_dir = tmp_path / "metadata"
    index_dir = tmp_path / "indexes"
    metadata_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)

    monkeypatch.setattr("app.retrieval.search.METADATA_DIR", metadata_dir)
    monkeypatch.setattr("app.retrieval.search.INDEX_DIR", index_dir)

    doc_id = "test-doc-overlap"
    # Create chunks where chunk 1 and chunk 2 have significant overlap
    chunks_data = [
        {
            "chunk_id": "chunk_0",
            "page_number": 1,
            "section": "General",
            "text": "Applicants must submit a completed application form and academic transcripts.",
            "bboxes": [],
        },
        {
            "chunk_id": "chunk_1",
            "page_number": 1,
            "section": "General",
            "text": "Applicants must submit a completed application form and academic transcripts. Proof of identity is also required.",
            "bboxes": [],
        },
        {
            "chunk_id": "chunk_2",
            "page_number": 2,
            "section": "General",
            "text": "Interviews may be conducted for shortlisted applicants in November.",
            "bboxes": [],
        },
    ]

    meta_file = metadata_dir / f"{doc_id}.json"
    meta_file.write_text(json.dumps({"document_id": doc_id, "chunks": chunks_data}), encoding="utf-8")

    # Fake faiss index
    faiss_file = index_dir / f"{doc_id}.faiss"
    faiss_file.write_text("fake_faiss_data", encoding="utf-8")

    # Mock faiss.read_index and embedding model
    class MockIndex:
        def search(self, emb, k):
            # return chunk indices [0, 1, 2]
            return None, [[0, 1, 2]]

    monkeypatch.setattr("faiss.read_index", lambda path: MockIndex())

    results = search_document(doc_id, "application form", top_k=3)

    # Chunk_0 is a complete substring of Chunk_1, so duplicate suppression should filter one out!
    chunk_ids = [r["chunk_id"] for r in results]
    assert not ("chunk_0" in chunk_ids and "chunk_1" in chunk_ids), (
        f"Redundant overlapping chunks should be deduplicated: {chunk_ids}"
    )

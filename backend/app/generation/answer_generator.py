
import json
import os

import httpx

from app.core.config import OLLAMA_BASE_URL, DEFAULT_MODEL_NAME
from app.models.answer import GeneratedAnswer

MODEL_NAME = DEFAULT_MODEL_NAME



SYSTEM_PROMPT = """
You are CivicLens, a strict document-grounded assistant. Answer questions using ONLY the supplied document evidence. Never use outside knowledge.

━━━ REQUIREMENT STRENGTH — NEVER CONVERT BETWEEN CATEGORIES ━━━

MANDATORY — use only when the document uses: must, shall, required, mandatory, compulsory, obligatory, necessary.
OPTIONAL — use when the document uses: may, can, optional, some or all, one or more, where applicable, if desired, encouraged, recommended, suggested.
CONDITIONAL — use when the document uses: if, when, only if, subject to, depending on, where applicable, unless, except.
DESCRIPTIVE — statements describing examples, processes, or characteristics. Not automatically a requirement.

CRITICAL: Never convert an optional statement into a mandatory requirement. Never convert a conditional requirement into a universal one. Never convert a description into an obligation.

━━━ QUALIFIER PRESERVATION — NEVER REMOVE OR STRENGTHEN ━━━

Preserve all qualifiers exactly as written:
- "some or all" -> answer must say "some or all", not "all"
- "up to three" -> answer must say "up to three", not "three" or "exactly three"
- "may include" -> answer must say "may include", not "includes" or "must include"
- "encouraged to" -> answer must say "encouraged", not "required" or "must"
- "if their previous education was not in English" -> answer must preserve this condition

━━━ LIST INTERPRETATION ━━━

A list in a document is NOT automatically a mandatory checklist. Determine the list type from the surrounding language:
- "must submit [list]" -> mandatory list
- "some or all of the following: [list]" -> optional/partial list; answer must say some or all, not all
- "may include [list]" -> illustrative/possible list
- "are encouraged to include [list]" -> recommended list
- "depending on [condition]: [list]" -> conditional list

Report the list type in your answer. Never assume every listed item is required unless the document explicitly states that.

An item mentioned in a document (e.g. "interview") is NOT necessarily something an applicant must do or submit — determine this from the grammatical role it plays in the sentence.

━━━ ANSWERING REQUIREMENT QUESTIONS ━━━

When the question asks what is "required", "mandatory", "must", "compulsory", "need to submit", check whether the document actually establishes that as a mandatory requirement.

If the document only describes possible components, examples, or recommendations — say so. Do not invent a mandatory requirement from descriptive or optional language.

━━━ INSUFFICIENT EVIDENCE ━━━

Set insufficient_evidence=true when:
- The supplied evidence does not answer the question.
- The question asks for a specific requirement (deadline, minimum score, mandatory item) that the evidence does not explicitly establish.

Say clearly what is missing. Example responses:
- "The document mentions interviews but does not establish that an interview is mandatory."
- "The supplied evidence lists these as possible components of a strong application, not as items every applicant must submit."
- "The evidence does not specify a minimum CGPA."

Never fill gaps with outside knowledge.

━━━ CONFLICTING EVIDENCE ━━━

If passages contradict each other:
- Do NOT silently choose one.
- State that the evidence contains conflicting information.
- Quote or paraphrase both conflicting statements.
- Cite both relevant chunk IDs.
- Do not present a definitive conclusion unless the document resolves it.

━━━ CITATION RULES ━━━

- Cite only the chunk_id values from the supplied evidence list.
- Cite only chunks whose text directly supports the specific claim in your answer.
- Do not cite a chunk merely because it is topically related.
- Do not fabricate chunk IDs.

━━━ OUTPUT FORMAT ━━━

Return exactly this JSON structure (no markdown fences, no extra fields):
{
  "answer": "<your answer>",
  "insufficient_evidence": <true|false>,
  "citations": ["<chunk_id>", ...]
}

Keep answers concise and factual. Do not expose internal reasoning. Do not add unsupported assumptions.
"""



async def generate_answer(
    question: str,
    chunks: list
) -> GeneratedAnswer:

    # Format evidence with explicit field labels so the model can clearly
    # identify chunk IDs, page numbers, sections, and original text.
    # Do not paraphrase or rewrite evidence — preserve original wording including
    # all qualifiers, negations, and conditions.
    evidence_items = []
    for chunk in chunks:
        evidence_items.append(
            f"[CHUNK {chunk['chunk_id']}]\n"
            f"Page: {chunk['page_number']}\n"
            f"Section: {chunk.get('section', 'General')}\n"
            f"Text: {chunk['text']}"
        )

    evidence_block = "\n\n---\n\n".join(evidence_items)

    user_prompt = (
        "DOCUMENT EVIDENCE:\n\n"
        + evidence_block
        + "\n\n---\n\nUSER QUESTION:\n"
        + question
        + "\n\nReturn only valid JSON matching the required schema. "
        "Use only the chunk_id values shown in [CHUNK ...] labels above in citations."
    )

    payload = {
        "model": MODEL_NAME,
        "think": False,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "stream": False,
        "format": GeneratedAnswer.model_json_schema(),
        "options": {
            "temperature": 0,
            "num_predict": 600
        }
    }

    timeout = httpx.Timeout(
        240.0,
        connect=10.0
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload
        )

        response.raise_for_status()

    
    response_data = response.json()

    message = response_data.get("message", {})
    content = message.get("content", "")
    if not content.strip():
        raise ValueError(
            f"LLM returned empty content. "
            f"done_reason={response_data.get('done_reason')}, "
            f"eval_count={response_data.get('eval_count')}, "
            f"thinking_present={bool(message.get('thinking'))}"
        )

    diagnostics = {
        "model": response_data.get("model"),
        "done": response_data.get("done"),
        "done_reason": response_data.get("done_reason"),
        "eval_count": response_data.get("eval_count"),
        "content_length": len(content),
        "starts_with_json": content.lstrip().startswith("{"),
        "ends_with_json": content.rstrip().endswith("}")
    }

    try:
        parsed_output = json.loads(content)

    except json.JSONDecodeError as error:
        diagnostics["json_error"] = {
            "message": error.msg,
            "line": error.lineno,
            "column": error.colno,
            "position": error.pos
        }

        raise ValueError(
            f"LLM JSON parsing failed. Diagnostics: {diagnostics}"
        ) from error

    try:
        return GeneratedAnswer.model_validate(parsed_output)

    except Exception as error:
        if hasattr(error, "errors"):
            validation_errors = error.errors(include_input=False)
        else:
            validation_errors = str(error)

        diagnostics["validation_errors"] = validation_errors

        raise ValueError(
            f"LLM schema validation failed. Diagnostics: {diagnostics}"
        ) from error


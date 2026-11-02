This test suite validates the Document Parser stage of a Legal Intelligence AI Agent.

Categories:
- HAPPY PATH: confirms the parser can read clean text documents, detect contract type, and return well-formed clause chunk objects.
- EDGE CASES: confirms failures are handled safely for empty/corrupt inputs and unsupported file types.
- VAGUE QUERY: confirms the parser triggers a clarification workflow for underspecified user requests and only returns intent labels for specific questions.
- ADVERSARIAL: confirms the parser does not hallucinate clauses that are not grounded in the document when retrieval fails or returns no evidence.

Known limitation — corrupt .docx detection:

`DocumentParser._load_text_from_path` does not use a real DOCX reader. For a
`.docx` input it calls `raw.decode("utf-8")` and raises
`ParseError("corrupt or malformed docx")` only when the bytes are undecodable.
Two consequences worth knowing before trusting `test_corrupt_docx_returns_error`:
- A "corrupt" file whose bytes happen to be valid UTF-8 text parses successfully,
  so the `corrupt_docx_path` fixture must contain invalid UTF-8 to exercise the
  failure path at all.
- A genuine `.docx` is a zip archive and is therefore also undecodable, so real
  DOCX input hits the same error. `.docx` is effectively unsupported despite
  being listed in the accepted suffixes and `python-docx` being in
  `requirements.txt`.

How to add new test cases:
- Add new fixtures under `tests/fixtures/` when you need realistic sample contract text.
- Add new test functions to `tests/test_document_parser.py` in the appropriate test class.
- If the test depends on query classification or retrieval, patch the injected LLM/embedder methods in `conftest.py` or override them in the test with `unittest.mock.patch`.

Semantic Embedder Tests

`tests/test_semantic_embedder.py` validates the Layer 2 vector retrieval component.

Categories:
- HAPPY PATH: embeds chunks, retrieves top-k similar chunks, validates output schema and `similarity_score` bounds.
- EDGE CASES: verifies correct behavior when embedding fails, when no chunks are embedded, and when `reset()` clears the index.
- RETRIEVAL QUALITY: checks that relevant queries rank higher than irrelevant ones, that totally unrelated queries yield low scores, and that re-embedding after reset prevents cross-document contamination.

Query Intent Classifier Tests

`tests/test_query_intent_classifier.py` validates Layer 3 query understanding before Agent Core execution.

Categories:
- INTENT CLASSIFICATION: checks mapping of specific legal questions to allowed intent labels (`termination`, `breach`, `payment`, `data`, `IP`, `employment`, `liability`, `confidentiality`).
- VAGUE DETECTION: verifies short and generic queries trigger clarification mode with suggested topics and no intent assignment.
- ENTITY EXTRACTION: validates regex-only extraction of time references, monetary amounts, party references, and contract type signals.
- EDGE CASES: validates long/special-character queries, required result keys, preserved `raw_query`, and bounded confidence scores.

Agent Core Tests

`tests/test_agent_core.py` validates Layer 4 orchestration and final legal analysis shaping.

Categories:
- ANALYZE: validates risk-level tagging, triggered-clause detection, and source-position grounding.
- TRANSLATE: validates plain-English rewriting, jargon cleanup, and source-position preservation.
- GAP CHECK: validates missing-clause detection against contract-type templates and output schema.
- RUN AND VALIDATE: validates 4-step orchestration, confidence filtering, count integrity, required output keys, and readable summary output.

Knowledge Base Tests

`tests/test_knowledge_base.py` validates Layer 5 static legal reference loading and gap-check logic.

Categories:
- Loading: frameworks and contract templates are present, parseable JSON, and match expected schema keys.
- Required clauses: contract types resolve case-insensitively and return template required clauses.
- Frameworks: each framework includes obligations with valid severities.
- Gap checking: missing clause detection returns correct clause types and `severity_if_missing` values, and unknown contract types raise errors.

Output Report Tests (Layer 6)

`tests/test_output_report.py` validates the final formatting layer that converts AgentCore results into clean user-facing outputs in `dict`, `text`, and `markdown` formats.

Categories:
- Risk and severity sorting is deterministic and correct.
- Disclaimer is always present and correctly formatted.
- Edge cases: empty clauses, empty missing clauses, and invalid input/format handling.

# Bound results and task artifacts

A first frozen real run passed 32/40 unique tasks and 46/56 trials. Its failures and scientific limitations remain recorded in EXPANDED_ACCEPTANCE_V6.md. The changes here are subsequent regression fixes, not replacement model-quality scores.

The application adds result_refs to the model-facing draw_curve/draw_table schemas. Each reference names a completed result_id, a numeric source path and a chart-data destination. The runtime resolves it before MCP schema validation, records actual resolved arguments and retains requested arguments/result_id in the trace. The actual MCP server receives ordinary resolved numeric data. Once the current task has computed results, following plots require references; independent input plots can run before computation or in a separate request. References are local to one request, reject nonfinite values, unknown IDs, overlapping targets and non-data destinations. This binds selected numeric values; it does not prove that a model selected the scientifically appropriate variable or chart.

Successful tool file_path values must resolve within the generated-result directories; images must decode. File fingerprints are recorded on completion and checked again before release. The final answer is buffered until task-local artifact checks pass. Structured artifacts must match this task's actual registered files. Files from prior tasks, fabricated index downloads, deleted or changed files and external artifact URLs are rejected before content is emitted or persisted as a successful answer. Other SSE events still carry task/tool progress. Index ingestion explicitly reports no downloadable artifacts.

```powershell
python scripts/verify_execution_contracts.py --output work/execution-contracts
```

This script uses a scripted provider and an actual local MCP STDIO subprocess, with no paid LLM calls. Five scenarios pass: computed value bound into an actual table; unbound derived data rejected; an existing file from another task rejected; invented ingestion download rejected; independent user data still renders. It checks persisted task outcomes and whether rejected text was emitted. It is not a real-model quality evaluation.

The current environment supports drawing, tables, fixture ingestion/retrieval, bandgap and legacy aging. Single/batch silicon simulation and perovskite parameter prediction require missing model assets or optional dependencies; these remain unavailable. Bandgap legacy families and aging scientific accuracy retain their existing limitations. No training data, coefficients or scientific model results were changed.

Backend-unavailable UI fallbacks now log ASCII events and exception types instead of printing localized exception bodies. Six startup handlers cover model-list, history, conversation-choice and solar/aging/perovskite default-parameter failures. Each has an isolated strict-CP1252 regression with no live network. The complete UI is also built under both UTF-8 and strict CP1252 with the backend forced unavailable. All seven added cases failed before their corresponding fixes and now pass; the full local suite passes 231 tests.

The same five actual local MCP subprocess scenarios also pass with PYTHONIOENCODING=cp1252:strict; this does not add real-provider trials.

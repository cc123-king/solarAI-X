# Expanded Agent acceptance v6

40 distinct author-written tasks: 24 positive, 16 negative. Eight selected tasks run three times; the remaining 32 run once, for 56 planned trials. The first frozen real-model run completed on 2026-10-19: 32/40 unique tasks and 46/56 total trials passed strict acceptance; 7/8 selected tasks passed all three attempts. It used 110 chat requests and no external embedding requests.

The suite uses the actual application stream, task persistence and an actual local MCP STDIO subprocess. The model sees the discovered nine-tool catalog. JSON results are checked against persisted tool arguments/results; PNG paths must match actual completed calls and files must decode. Transport errors cannot pass as model-unavailable business errors. SDK retries are zero, with a persistent limit of 224 chat requests, zero external embedding requests, three tool rounds and eight tool calls per trial. Inputs for directory ingestion are limited to two fictional fixture documents under evaluation/fixtures_v6/source. Retrieval uses the lexical fallback; it does not measure BGE semantic retrieval quality.

The current dependency profile can execute drawing, table generation, fixture ingestion/search, bandgap and legacy aging. Solar single/batch simulation and perovskite parameter prediction currently lack model assets or optional dependencies; their cases verify explicit unavailable behavior. Legacy aging is checked for execution/output consistency only, without a new scientific accuracy claim. Public measured bandgap estimates stay inside the original model neighbourhood; out-of-domain inputs remain rejected.

Preparation exposed two protocol defects: embedded Image objects prevented aging output from serializing through MCP, and progress prints polluted STDIO. Aging now returns structured values and its generated PNG path; progress goes to stderr. Out-of-domain bandgap errors have their own code instead of being collapsed into generic invalid input. Original model coefficients and scientific results are unchanged.

```powershell
python scripts/evaluate_agent_expanded.py --preflight work/solar-preflight.json
python scripts/evaluate_agent_expanded.py --oracle work/solar-preflight.json --prepare-scope work/solar-scope.json
# Review the frozen questions, inputs and budget before enabling provider calls.
python scripts/evaluate_agent_expanded.py --oracle work/solar-preflight.json --scope-manifest work/solar-scope.json --allow-live-calls --output work/solar-live
```

Preflight uses real local tools and zero paid model calls; it is not model decision-quality evaluation. Once the first model run begins, keep questions, grading, model configuration and prompts frozen until its complete report is recorded. Report 40 unique tasks separately from 56 trials; summarize the eight three-trial consistency results separately. Preserve failures and interruptions. Do not rewrite original 12-task results or claim general task accuracy.

## First frozen results and offline follow-up

Positive tasks passed 20/24; negative handling passed 12/16. Ten failed trials include one null value passed to a derived table, four exact-identifier retrieval/contract failures, one invented ingestion artifact path, and four unavailable/error status mismatches. Keep these original failures. The preflight had accepted zero-result searches, so 40/40 tool preflight did not establish positive retrieval semantics.

After preserving frozen source and results, terminal-period tokenization, structured unavailable feedback, and rejection of null/nonfinite/ragged table data were corrected. Twenty regression cases were added: 198 full Windows tests and eight real local MCP checks pass. No paid model retest of these changes has run. Ingestion artifact reporting and general multi-step result fidelity remain follow-up work. These changes have not yet been published or checked by new Ubuntu CI.

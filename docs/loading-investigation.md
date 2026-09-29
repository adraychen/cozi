# Loading slowdown investigation — 2026-09-29

## Finding

The reported production slowdown is **not yet reproduced or conclusively attributed**. The Select all/Clear all session-state fix does not show a material server-side slowdown in a controlled local comparison. The previous explanation identified longstanding bottlenecks but did not establish the cause of a recent regression.

No application code or layout was changed during this investigation. The earlier uncommitted loading optimizations remain in the working tree; they were excluded from the historical comparison.

## Exact changes examined

- `34264a6`: before the native-table and per-group predictive bulk-button changes.
- `5cb632d`: includes those changes, before the reported Streamlit button-state error was fixed.
- `6309323`: the error fix. Its only application change is filtering the new button-key prefixes and filtering restored dataset state. The remainder of this commit adds tests. It changes no model, chart calculation, network call or training trigger.

Historical files were extracted to isolated temporary directories. Working application files were not reverted or swapped.

## Controlled measurements

Same Python environment and deterministic synthetic data: 1,000 rows, 16 columns (10 numerical, four categorical, two dates), seed 420. Full app execution through Streamlit AppTest. Database statistics, visit recording and AI responses were mocked; no production data, Supabase queries, training or real AI calls were used. Each revision ran in a separate process, sequentially, with cProfile enabled during app execution.

For each revision: three independent sessions for Inspect, then three for Analyze. The first execution includes cold imports; the next two reuse process-level imports/caches but use new sessions. Therefore “repeat mean” is not an existing-session interaction measurement. Actual workflow state and absence of exceptions were checked. These are approximate local **server/AppTest** times, including profiling overhead; they are not user-visible browser paint measurements or statistically powered performance estimates.

| Revision | First Inspect (s) | Repeated Inspect mean (s) | First Analyze (s) | Repeated Analyze mean (s) | Interactive tables in Inspect |
| --- | ---: | ---: | ---: | ---: | ---: |
| `34264a6` — Before table/control changes | 4.12 | 1.56 | 3.40 | 0.70 | 2 |
| `5cb632d` — After table/control changes, before button-state fix | 4.11 | 1.54 | 3.41 | 0.65 | 20 |
| `6309323` — After button-state fix | 4.07 | 1.54 | 4.41 | 0.66 | 20 |

The immediate before/after-fix repeated Analyze difference is approximately 0.01 seconds. First-load times vary; a single cold sample cannot establish a regression. In the after-fix first Analyze profile, importing the modelling module/SHAP accounted for approximately 2.23 seconds of 4.15 seconds in `render_analyze`. Those imports existed in every compared revision.

An isolated five-repeat microbenchmark of the old/new key predicate over 1,002 representative keys measured approximately 0.183 versus 0.191 milliseconds per complete scan. This does not benchmark browser state transport, but shows the added prefix checks themselves do not explain a seconds-long delay. Existing button values are now correctly excluded.

## Remaining suspects and evidence limits

1. **Browser table rendering:** Inspect now produces 20 native interactive tables instead of two for this dataset, mainly from the quality-summary panels and Column Overview. The change is real and coincides with the broader UI update. Server time did not increase materially; additional browser painting/memory cost has not been measured. It must not be presented as the confirmed cause.
2. **Cold deployment/startup:** heavy imports can delay the first Analyze visit. They predate the fix. A restart could make a previously warm app seem slower, but no deployment/restart evidence was available.
3. **AI service latency:** every historical fresh Analyze session made one mocked interpretation call before Predictive Analytics controls were reached. A live slow request would delay those controls. Automatic calls predate the fix, and production request timings were not available.
4. **Dataset and existing results:** dataset size/type mix, temporal parsing, chosen variables and previously trained SHAP results can affect rerun cost. The user's exact data, selection state and timing scenario were not supplied, so this benchmark does not rule out a data-dependent regression.

## Next measurement needed

Reproduce with the affected dataset (or a synthetic equivalent), the relevant saved selections/results, and the same deployed version. Distinguish initial upload, first Analyze visit, predictor changes and Run/training. Measure server stages alongside browser rendering/network timing. If server timings are stable but the page paints slowly, focus on the native table change; if Predictive Analytics waits on an interpretation request, measure that request; if only first visits are slow, inspect cold-start/import timing. Do not redesign navigation as a substitute for identifying the regression.

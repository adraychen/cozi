# Temporal target guidance

## Routing and validation

Predictive Analytics checks the effective target classification before rendering predictor or model controls. Numerical outcomes use regression, binary categorical outcomes use binary classification, and other categorical outcomes use multiclass classification. Temporal outcomes open guidance and return immediately. Numeric 0/1 outcomes remain numerical unless the user explicitly classifies them as categorical.

All six predictive training entry points reject raw datetime/timedelta targets and inferred date-like text. The shared metadata router also returns a dedicated temporal route. Explicit categorical overrides remain supported. No datetime target is converted to a numeric timestamp and no derived duration column is created.

The target checkbox group includes all temporal subtypes, including Time, Duration, Year, and Month, rather than only calendar dates. Target selection and guidance remain scoped to the dataset through existing workspace persistence. The new action button is excluded from saved widget state.

## Guidance and AI

Deterministic candidate discovery uses the existing temporal parser and effective metadata:

- Compatible calendar reference dates with at least one observed pair and no negative observed interval can suggest Time-to-Event Analysis under Reliability Analysis.
- A likely binary event-status column plus reference dates or a nonnegative numerical lifetime/duration column can suggest Survival Analysis under Reliability Analysis.
- Time Series Forecasting is considered only after the user explicitly chooses forecasting over time and a numerical measurement, with a supported calendar subtype and at least three paired observations.
- Choosing a numerical or categorical target remains the standard available path.

These checks identify possible analytical questions, not sufficient validation to fit a specialist model. Column meanings, event-status coding, observation endpoints, and analytical assumptions still need confirmation. Missing endpoints are never silently filled. A duration suggestion is explanatory only.

The existing Groq client receives column names, effective types/subtypes, the user's selected goal, and validated candidate IDs. It receives no raw records. The prompt requests concise JSON interpretation and reasoning, identifies specific analysis names, and prohibits automatic transformations, model execution, timestamp conversion, and treating dates alone as forecasting.

The app validates the returned candidate ID and explanation fields. Invalid JSON, an unsupported recommendation, missing credentials, service failures, and a 20-second request timeout produce deterministic fallback guidance. Responses are cached by metadata/goal/candidate context to avoid repeated calls on ordinary reruns. AI-generated prose is displayed as plain text and cannot define actions.

## Availability and assumptions

The current app has no Time-to-Event Analysis, Survival Analysis, or Time Series Forecasting page. Recommendations explicitly identify these as unavailable. Only **Choose Another Target** is actionable; it selects the first available numerical or categorical column and resumes the existing standard workflow. If none exists, the action is disabled. Adding specialist models or automatically deriving durations is outside this change.

Reference dates and binary status names are clues, not proof of a delivery or failure process. The user can clarify their goal before acting. Caching uses the metadata/candidate summary rather than raw values, because the AI prompt itself is metadata-only.

## Files changed

- `utils/predictive_routing.py`: shared temporal detection and training validation.
- `utils/temporal_target_guidance.py`: deterministic candidates, fallback, response validation, cached guidance UI, and target-switch action.
- `utils/ai_interpreter.py`: metadata-only structured recommendation prompt and bounded service request.
- `utils/dataset_views.py`: intercept temporal targets before normal predictive controls.
- `utils/modelling_config.py`: deterministic target routing, including numeric outcomes without subtype metadata.
- `utils/variable_metadata.py`: temporal metadata target route.
- `utils/modelling.py`: guards in all six training entry points.
- `utils/predictive_controls.py`: include every temporal subtype in the target group.
- `utils/dataset_workspace.py`: exclude the new button from persistent widget state.
- `tests/test_temporal_target_guidance.py`: routing, immutable raw dates, delivery/failure/forecast cases, invalid AI responses, service fallback, training guards, UI, available actions, and workspace persistence.
- `tests/test_predictive_controls.py`: grouped single-target selection and retained selection.
- `tests/test_analyze_loading.py`: recognize target checkboxes and verify existing lazy analytical behavior.
- `README.md`: target routing and guidance documentation.

The Enterprise branding color change already present in `utils/branding.py` belongs to the preceding request.

## Manual verification

A fresh local browser session used synthetic Order Date, Delivery Date, Sales, and Product columns. Selecting Delivery Date showed Time-to-Event Analysis and the conceptual Delivery Date − Order Date relationship. No standard model controls or runnable reliability action appeared. Choosing a forecasting goal and Sales showed Time Series Forecasting guidance. Choose Another Target returned to numerical Sales and the existing regression setup. No model training was triggered.

The live AI service was unavailable in this environment, so the browser exercised deterministic fallback. Successful structured AI responses and invalid-response handling were exercised with mocks in automated tests. The browser check also caught and verified the fix for saving the action button as persistent state. Visual evidence is in `temporal-guidance.png`.

The final full automated suite passed: **250 tests in 165.255 seconds**, with no failures or errors. Full results are recorded in `../temporal-test-results.log`. `git diff --check` passed. The Groq timeout configuration was also verified against the installed client without making an external request.

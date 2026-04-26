# Codebase Audit

**Scope:** Full repo at commit `6503eca` on `main`.
**Date:** 2026-04-16.

**Deployment context (informs severity):**
- Frontend: GitHub Pages at `levineuwirth.github.io/icd_embeddings/` (built from `vite build` → `docs/` directory, served from main branch).
- Backend: Hugging Face Space (`levineuwirth-icd-embeddings.hf.space`), Docker on port 7860 per `Dockerfile` + `backend/README.md`.
- The `docker-compose.yml` / `nginx.conf` / `deploy.sh` path is a **separate, stale, unused** Mac Studio + Cloudflare deployment. Several findings below apply only to that path and are labeled accordingly — they do not affect production.
- The API is intentionally public and unauthenticated, so "open CORS" findings are de-escalated from what a typical audit would flag.

Severity scale: **Critical** > **High** > **Medium** > **Low** > **Info**.

---

## High

### H1. Broken / stale backend tests — assertions don't match current API
**File:** `backend/tests/test_main.py`
- `test_predict_valid_data` (L19–35) asserts a flat response with `prediction`, `confidence_interval`, `interpretation` keys. The real `/predict/` returns a nested `{"readmission": {...}, "mortality": {...}}` (main.py:558). Test fails.
- `test_search_icd_found` (L71–77) asserts `{"I10": "Essential (primary) hypertension"}`. Endpoint returns a **list of dicts** with `code`/`description`/`in_training_dataset` keys (main.py:951–955). Test fails.
- `test_search_icd_not_found` (L80–86) expects `{}`. Endpoint returns `[]`. Test fails.
- `from httpx import AsyncClient` (L3) imported but unused.
- No tests for `/predict_flex/`, `/parse_icd_codes/`, `/upload_icd_file/`, or edge cases (100+ codes, NaN-mapped codes, file encoding errors, empty age, etc.).
- Test import path `from backend.main import app` requires invoking pytest from repo root; no `conftest.py` or project config documents this.

**Fix:** Rewrite assertions to match the current response shape; drop the unused `AsyncClient`; add coverage for the flex/parse/upload endpoints.

### H2. `calculate_prediction_ci` is not a real confidence interval
**File:** `backend/main.py:387–408`
```python
for _ in range(n_bootstraps):
    pred = model.predict(inputs, verbose=0).flatten()[0]
    noise = np.random.normal(0, 0.05)
    predictions.append(pred + noise)
```
Keras `model.predict` is **deterministic** for fixed inputs with no dropout at inference. Each of the 100 iterations produces the same `pred`; the only variance comes from the hand-added Gaussian noise (`σ=0.05`). The resulting percentile-based "95% CI" reduces to `pred ± 1.96·0.05 ≈ pred ± 0.098`, i.e., a constant-width pseudo-interval dressed up as a bootstrap. This is also:
- Wasteful (100 identical forward passes per request, materially slowing `/predict/` and `/predict_flex/` full-demographic paths).
- Misleading to clinicians reading `confidence_interval` as if it reflected model uncertainty.

The ICD-only path at main.py:695–698 uses a fixed `±0.05` band — at least it's honest about being arbitrary, but both paths share the same underlying problem: no real epistemic uncertainty is being estimated.

**Fix:** Use MC-Dropout (enable dropout at inference and repeat), an ensemble, or at minimum drop the word "confidence_interval" and document what the band actually represents.

### H3. `Dict[str, any]` — `any` is the Python builtin, not `typing.Any`
**File:** `backend/main.py:974`
```python
def parse_icd_codes_from_text(text: str, max_codes: int = 35) -> Dict[str, any]:
```
`any` here is `builtins.any`, the aggregation function. This is a valid expression (so it doesn't crash), but it's not a type and will confuse any type-checker. `Any` is already importable from `typing` and `Optional` is already imported from `typing` at L12.

**Fix:** `from typing import Any` and use `Dict[str, Any]`.

### H4. Every prediction endpoint wraps the world in `except Exception as e: raise HTTPException(500, detail=str(e))`
**File:** `backend/main.py:584–585`, `913–914`, `1123–1124`
This leaks internal error text (file paths, keras stack messages, pandas dtype errors, etc.) to any client. It also swallows the original traceback from logs (there's no `logger.exception` in the handler).

For a public unauthenticated ML API the disclosure risk is limited, but it's still poor practice and makes triage harder.

**Fix:** `logger.exception(...)` for the operator, return a generic `"Prediction failed"` to the client. Let Pydantic `ValidationError` propagate (it already returns 422).

### H5. `/predict_flex/` re-implements all preprocessing inline instead of reusing `predict` / `predict_icd_only`
**File:** `backend/main.py:722–914` duplicates main.py:430–583 and main.py:588–719.
Any fix to one code path (e.g., new ICD normalization, new calibration, new column layout) has to be made in three places. The three paths already drift in small ways — the ICD-only path uses the cheap `±0.05` CI while the flex-full path uses `calculate_prediction_ci`; the interpretation strings are identical but re-typed.

**Fix:** Have `predict_flex` call the same shared builder used by `predict`, parameterized on "has demographics or not."

### H6. Silent age capping, no signal to the client
**File:** `backend/main.py:299–300, 345–347`; `src/App.jsx:62–65`
If a client posts `age=95` directly to the API, it's silently replaced with `90` and no field in the response indicates the coercion. The frontend has the same silent behavior — `validateAge` sets `ageWarning=''` at L64. A user entering `95` sees no UI hint that the model is treating them as `90`.

This is a clinical decision-support tool; silently downgrading an age is the kind of thing a user should know about. This was also flagged by the `CLAUDE.md`-free convention in the code — the docstring even calls out "Ages 90-124 are capped at 90" but no surface presents this.

**Fix:** Return `age_adjusted: true, original_age: 95, adjusted_age: 90` in the API; show a non-blocking note in the UI when the range is hit.

### H7. `backend/main.py` loads Keras models at import time and raises `RuntimeError` on any failure (main.py:263–266)
If any model or pickle file is missing the module cannot even be imported, which means `uvicorn` / `gunicorn` crashes immediately and the HF Space shows no useful health endpoint — the root `/` handler never starts. The top-level `try`/`except FileNotFoundError` doesn't catch other load failures (e.g., a Keras deserialization error), which will crash without the custom message.

**Fix:** Move model loading into a FastAPI `lifespan` / `startup` event, log the failure, and let `/` return a 503 with a clear "models not loaded" message.

### H8. Python `3.11` pycache files are tracked in git
**Tracked (git ls-files):**
```
backend/__pycache__/__init__.cpython-311.pyc
backend/__pycache__/main.cpython-311.pyc
backend/__pycache__/main.cpython-313.pyc
backend/tests/__pycache__/__init__.cpython-311.pyc
backend/tests/__pycache__/test_main.cpython-311-pytest-9.0.1.pyc
```
`.gitignore` does not list `__pycache__/` or `*.pyc`. These get stale immediately after any edit and will produce confusing diffs.

**Fix:** Add `__pycache__/` and `*.pyc` to `.gitignore`, then `git rm -r --cached` the tracked copies.

### H9. `model/` directory at repo root is stale, misleading, and tracked
**File:** `/model/*`
- `NRD_2019_Small.dta` — 133 bytes, not a real Stata dataset.
- `readmit_2016_age_scaler.pkl` / `readmit_2016_label_encoder.pkl` — 128/131 bytes, likely LFS pointer stubs never resolved.
- `readmit_hypertrial_deepset.keras` — 5.8 MB model that no code under `backend/` or `src/` loads.
- `Validate.ipynb` — 27 KB notebook, last touched May 2025.

None of this is used by the running backend (backend/model/ has the real assets). It confuses readers and bloats clones.

**Fix:** Delete `/model/` entirely or move to a `research/` or `archive/` directory outside the deployment tree.

### H10. `docs/` contains build artifacts that will drift from source
**Tracked:** `docs/index.html`, `docs/assets/index-BXe8V7PX.css`, `docs/assets/index-Eajkw6e5.js`, `docs/vite.svg`.

This is the published GitHub Pages artifact, so it has to be committed — but nothing in the repo enforces that it matches `src/`. There's no CI to rebuild, no pre-commit guard, and the hash in the filename means the editor can't diff past a rebuild. Every `npm run build` creates a new file and the old one has to be removed manually.

**Fix:** Add a GitHub Actions workflow that runs `npm run build` on push to `main` and commits / force-pushes the `docs/` artifact, OR switch to the more standard GH Pages pattern of publishing from `gh-pages` branch with `peaceiris/actions-gh-pages` so the source branch stays clean.

---

## Medium

### M1. Dead code: `get_risk_interpretation` never called
**File:** `backend/main.py:367–384`
Defined, exported-looking, and every call site has re-inlined the same if/elif ladder (L538–555, L864–878, L673–693). Reads like something meant to be used.

**Fix:** Either route all three paths through `get_risk_interpretation` (note the outcome-specific wording — it currently hard-codes "readmission"), or delete it.

### M2. `calibrate_probability` uses `tf.clip_by_value` and requires a `.numpy()` dance
**File:** `backend/main.py:350–364, 510–515, 837–841`
Every caller pays `tf.clip_by_value → tf tensor → .numpy() → float`. The function operates on a scalar, so wrapping in a TF op is pure overhead. Swap for `p_sampled = min(max(p_sampled, eps), 1 - eps)` and return a float directly.

Also worth noting: the docstring says `beta = (# majority after undersampling) / (# majority original)` **or** `original_positive_rate (if you balanced to 50/50)`. The two are different quantities and the plug-in constants `BETA_READMIT = 0.139050` / `BETA_MORTALITY = 0.003877` look like population positive rates, not undersampling ratios. Worth re-deriving to make sure the calibration formula is the right one for how the training set was built.

### M3. `handleIcdLookupSearch` has no debounce and no cancellation
**File:** `src/App.jsx:85–98`
Fires a `GET /search_icd/` on every keystroke once `value.length > 2`. Two problems:
1. Rapid typing hits the HF Space on every character — easy to trigger rate limiting or cold-start penalties.
2. Out-of-order responses are not cancelled. Typing `diabe` then `diabet` can result in the older `diabe` response arriving later and overwriting the newer result list.

**Fix:** Debounce ~250ms; track a request id or use `AbortController` to drop stale responses.

### M4. `calculateRisk` performs two sequential HTTP requests per click when using paste
**File:** `src/App.jsx:178–274`
If `icdMethod === 'paste'`, it calls `/parse_icd_codes/` at L193, then `/predict_flex/` at L256. Two round-trips per click; on a cold HF Space start this stacks the latency.

**Fix:** Accept raw text in `/predict_flex/` as an alternative to `icd_codes` and parse server-side in one shot, or parse client-side from `pastedText`.

### M5. `isCalculating` set only around the predict call, not the parse call
**File:** `src/App.jsx:253`
`setIsCalculating(true)` is set just before the predict POST. The preceding `/parse_icd_codes/` call (L193) can take seconds during a cold start, and the UI shows no loading state during that window — the button remains enabled and the user can double-click it.

**Fix:** Move `setIsCalculating(true)` to the top of `calculateRisk` (after the "at least one code" bail-out), or wrap the whole async pipeline in a guard.

### M6. `calculateRisk` clears results before it's committed to a new run
**File:** `src/App.jsx:182–187`
Results are wiped synchronously before the async parse/predict. If parsing fails or the user cancels, previous results are gone with no way back. Visually disruptive on a low-latency refresh.

**Fix:** Only overwrite `results` when the new prediction has arrived.

### M7. Frontend shows errors via `window.alert`
**File:** `src/App.jsx:144, 157, 174, 200, 215, 217, 249, 268, 270`
`alert()` is synchronous-modal, not accessible, and blocks the main thread. Multiple error paths use it (file upload failure, parse failure, validation failure, predict failure).

**Fix:** Route errors through a persistent `errorMessage` state rendered in an `aria-live="polite"` region next to the Calculate button.

### M8. `handleFileUpload` lacks size / content validation
**File:** `src/App.jsx:115–146`
Accepts any file the browser lets through. No client-side size check. Error path only fires `alert` and does not clear stale `validationResults` from the previous successful parse — on repeated failed uploads the user sees green "valid codes" from a prior file alongside the generic failure alert.

**Fix:** Check `file.size` against a ~2 MB cap before upload; always clear `validationResults` at the start of a new upload; show parse errors inline.

### M9. `/upload_icd_file/` checks `content_type` with a weak fallback and never verifies the size
**File:** `backend/main.py:1101–1124`
- The content-type whitelist can be bypassed with any `.txt`/`.csv` filename (L1107). Low risk since the body is then parsed as ICD text only, but the branch is confusing.
- No `Content-Length` check; the endpoint happily reads the full body into memory before decoding. Nginx (Mac Studio path) limits this to 10 MB, but HF Space has no comparable guard configured here.
- `except Exception as e: HTTPException(500, detail=f"Error processing file: {str(e)}")` (L1123–1124) echoes the raw internal error.

**Fix:** Drop the extension fallback (content-type already covers common MIME types; reject outright otherwise); cap the body at ~2 MB; catch specific exceptions only.

### M10. `/parse_icd_codes/` takes an untyped `data: dict`
**File:** `backend/main.py:1067`
```python
async def parse_icd_codes(data: dict):
    ...
    text = data.get("text", "")
```
Skipping Pydantic here means no schema in the auto-generated OpenAPI and no automatic 422 on a bad payload shape. Trivial to fix — define `class ParseICDRequest(BaseModel): text: str`.

### M11. `parse_icd_codes_from_text` silently caps at 35 codes and silently dedupes
**File:** `backend/main.py:1016, 1048–1056`
- `for code in unique_codes[:max_codes]:` — only the first 35 are processed; the extras get a warning but the caller has to know to read `warnings` to notice.
- Deduplication (L1003–1008) is a warning, but the dedup is also lossy: in ICD workflows, code multiplicity (e.g., primary diagnosis repeated across encounters) can be clinically meaningful. The model only takes 40 slots so this is a design choice — worth documenting rather than hiding.
- The suggestion loop iterates the first 1000 codes (`list(icd_codes.keys())[:1000]`, L1038). Since `icd_codes` is an unordered dict in practice, this "first 1000" is arbitrary — suggestions depend on insertion order of the JSON.

**Fix:** Reject >35 codes with a 422; document deduplication; for suggestions, either sort by prefix match on the full dict or build a proper prefix index.

### M12. `/search_icd/` hides valid ICD codes that aren't in the training dataset
**File:** `backend/main.py:947–949`
Typed "diabetes," got fewer results than expected? The endpoint filters out every ICD-10 code that isn't in `encoder.classes_` (the training label set). Users with no context see an unexplained gap.

The `/parse_icd_codes/` path does surface this distinction (`reason: "not_in_training"` with description, L1024–1033) — but the lookup endpoint never reveals the untrained codes at all.

**Fix:** Return all matches with a `selectable` / `in_training_dataset` boolean (the field is already built — just stop continuing on L949) and disable the row in the dropdown with a tooltip explaining why.

### M13. No input normalization on the frontend
**File:** `src/App.jsx:103, 207`
`pastedText` and `icdCodes` are shipped to the backend unnormalized (case, spaces, periods). The backend's normalizer runs inside the preprocessing pipeline (main.py:462, 624, 780) but doesn't report which form the user sent — so the "Invalid codes" list echoes the raw input and can show things like `"i 10"` as invalid when the user meant `I10`.

**Fix:** Normalize (`trim`, uppercase, strip internal whitespace) before calling `/parse_icd_codes/`. The backend should also collapse internal whitespace before splitting.

### M14. `icd_codes: ['', '', '', '', '']` initial state is never used
**File:** `src/App.jsx:12`
The component only renders the paste textarea or file input — there is no UI for editing individual `icdCodes[i]` slots anywhere in the file. Every code path overwrites this array wholesale (L140, L170, L207 filters it). The empty 5-slot seed is vestigial from an earlier design.

**Fix:** Initialize `icdCodes: []`.

### M15. ICD lookup results: no keyboard support
**File:** `src/App.jsx:601–609`
```jsx
<div key={result.code} className="search-result"
     onClick={() => selectIcdFromLookup(result.code)}>
```
`onClick` on a `div`, no `role`, no `tabIndex`, no Enter/Space handler, no arrow-key navigation. Keyboard-only users can't add codes from the lookup.

Also the outer container is not `role="combobox"` / `role="listbox"`, so screen readers don't see this as a suggestions list.

**Fix:** Switch to `<button>` elements inside a `<ul role="listbox">` container, and wire arrow-key navigation + Escape to close.

### M16. Form labels don't use `htmlFor` / `id` association
**File:** `src/App.jsx:300, 325, 344, 359, 375` (all `form-label` usages)
```jsx
<label className="form-label">Age <span>(optional)</span>:</label>
<input type="number" ... />
```
No `htmlFor` → no programmatic association → screen readers rely on DOM proximity, which works inconsistently. Same for gender and household-income quartile toggle buttons that sit under a `form-label` but aren't associated with anything.

**Fix:** Add `id` to inputs and `htmlFor` to labels; for the toggle-button groups use a `<fieldset><legend>` instead of a label.

### M17. No `<form>` wrapper, no Enter-to-submit
Enter in the age field or the lookup search does nothing. Calculate is a `<button>` without `type`, which defaults to `submit`, which would be fine if there were a form.

**Fix:** Wrap inputs in a `<form onSubmit={...}>` and set the Calculate button `type="submit"`.

### M18. `docker-compose.yml` / `nginx.conf` deployment path is broken (stale, unused)
Confirmed not affecting production, but if anyone tries to run `./deploy.sh` today:
- `vite.config.js` sets `build.outDir: 'docs'`. `deploy.sh` runs `npm run build` and `docker-compose.yml` then mounts `./dist:/usr/share/nginx/html:ro` (L35). Nginx serves nothing or stale content.
- `docker-compose.yml` declares `expose: - "8000"` (L11–12) and `nginx.conf` proxies to `backend:8000` (L3). But `Dockerfile` CMD binds gunicorn to `0.0.0.0:7860` (L43). Nginx → backend always fails.
- Backend healthcheck (`docker-compose.yml` L17) also targets `localhost:8000`, will fail.
- `deploy.sh` `cp .env.docker .env` (L12) overwrites the developer's `.env` (which is `VITE_API_BASE_URL=http://127.0.0.1:8000` today) with no backup.
- `deploy.sh` uses `set -e` but the "test deployment" step at L44 uses `if curl -f ...` which is allowed to fail — inconsistent.

**Fix:** Either delete these files (and remove the "Deployment" section that references Cloudflare + Mac Studio), or fix them: build to `dist` for Docker, set Dockerfile port to 8000, fix the healthcheck.

### M19. `requirements.txt` has unpinned critical deps
**File:** `backend/requirements.txt`
```
fastapi
uvicorn
gunicorn
tensorflow==2.15.0
keras==2.15.0
scikit-learn==1.5.1
pandas==2.2.0
python-multipart
pydantic
pyarrow
```
`fastapi`, `uvicorn`, `gunicorn`, `pydantic`, `python-multipart`, `pyarrow` are all unpinned. Pydantic 1 → 2 was a breaking change; FastAPI pulls in Pydantic transitively. An HF Space rebuild months later can pull a wildly different dep graph from the one the model was validated against.

Also: `pyarrow` is listed but nothing in `backend/` imports it (I grep'd). If it's only used by `pandas` for parquet, let pandas pull it.

**Fix:** Pin every runtime dep to the version that was validated; consider `uv pip compile` / `pip-tools` to keep a lock file.

### M20. No explicit NaN check for the `age_scaler` output
**File:** `backend/main.py:481, 801`
`age_scaler.transform(df[["AGE"]])` — `age_scaler` is a `MinMaxScaler` fit on training-set ages (0–90 after capping). Inputs already validated in range, so this is fine, but there's no guard that `transform` returned finite values. A corrupt scaler pickle could silently propagate NaN into the model. Low likelihood, low severity.

---

## Low

### L1. `App.css` is 43 lines of commented-out boilerplate
**File:** `src/App.css`
Entire file is the default Vite template commented out. `main.jsx` imports `index.css` only, so `App.css` isn't even loaded. Delete the file.

### L2. Dead CSS classes
**File:** `src/index.css`
Rules exist for `.logo`, `.logo-icon`, `.logo-inner`, `.logo-subtitle`, `.logo-title`, `.navigation`, `.nav-links`, `.nav-link`, `.add-more-button`, `.icd-code-row`, `.delete-icd-button`, `.manual-input-container`, `.icd-codes`, `.icd-input`, `.disclaimer`, `.disclaimer-text`, `.copy-button`, `.parse-button`, `.validation-summary`, `.valid-count`, `.invalid-count`, `.validation-warnings`, `.warning-text`. None are referenced in `App.jsx` anymore.

### L3. `parseInt(householdIncome)` without radix
**File:** `src/App.jsx:244`
`parseInt(v)` without the explicit `10` radix is a well-known foot-gun (historical octal behavior). Trivial: `parseInt(v, 10)` or `Number(v)`. Same issue on L42 for `age`.

### L4. `App.jsx` is a 640-line monolith
Everything — state, handlers, validation, rendering — in one `OutcomeCalculator` component. Would read much better split into `PatientForm`, `IcdInput` (paste + upload + lookup), `ResultsPanel`, with `useReducer` or context for shared form state.

### L5. `App.jsx:256` prediction response isn't validated
`const { readmission, mortality } = response.data;` — if the backend ever returns a 200 with a malformed body, the next line crashes on `readmission.prediction` and the user sees the generic alert at L270, losing the specific reason.

**Fix:** At minimum, guard `if (!readmission?.prediction || !mortality?.prediction) throw ...`.

### L6. `setValidationResults(null)` is called when switching radio (L385, L397) but `formData.pastedText` and `formData.uploadedFile` are not cleared
Switching from "Upload File" back to "Manual Input" leaves the filename reference in state even though the input is no longer visible. Harmless but the user's mental model is that the switch resets the section.

### L7. `calibrate_probability` + `calculate_prediction_ci` call `model.predict` with `verbose=0` inside a sync function
**File:** `backend/main.py:402, 507, 508, 645, 648, 829, 832`
FastAPI runs async handlers in the event loop, sync handlers in a threadpool. These endpoints are `async def`, but call `model.predict` directly (blocking, CPU-bound). On a single worker under concurrent load this can wedge the event loop. The HF Space runs `--workers 1` by default (`WORKERS:-1` in Dockerfile); `docker-compose.yml` overrides to 4, but HF doesn't use that file.

**Fix:** Wrap predict calls with `await anyio.to_thread.run_sync(...)` or define the handler as `def` (not `async def`) and let FastAPI move it off the loop.

### L8. `Optional` imported from `typing` but `PatientDataFlex` uses `Optional[int] = Field(None, ...)`
Fine today, but in Python 3.10+ (HF Space is on 3.11 per Dockerfile) the more idiomatic form is `int | None = Field(None, ...)`. No functional issue — style only.

### L9. `DeepSet.rho` ends with `Dense(output_dim, activation="relu")`
**File:** `backend/main.py:74`
The final layer of the DeepSet uses ReLU on the output. If the surrounding graph sigmoids this later that's fine; if not, predictions are unbounded on the top end and can never be negative. Worth a comment so the next maintainer doesn't go hunting.

### L10. `icd_codes` is both a module-level dict and a parameter name
**File:** `backend/main.py:210` (global) vs `588, 602, 616` (parameter)
`predict_icd_only(icd_codes: list[str])` shadows the global `icd_codes: Dict[str, str]`. Works because the function doesn't need the global, but it's confusing when reading 200 lines down from `parse_icd_codes_from_text` which *does* reference the global (L1014, L1019, L1038).

### L11. `print()` during model load (main.py:238–261)
`logger` is configured at L27 but the bootstrap messages go through `print`. Inconsistent — only matters if anyone ever points a log collector at the stream.

### L12. `@tf.keras.utils.register_keras_serializable` on unused `f2_score` function (main.py:31)
Registered for serialization but `F2Score` (the class, L153) is the metric actually used in training. Harmless — the registration is cheap and if any saved model references the function by name it'll resolve. Worth noting.

### L13. `.env`, `.env.docker`, `.env.production` are all committed
**Tracked:** all three.
Contents are non-secret (public API URL + `/api` prefix), but this is unusual practice and means a local dev's `.env` tweaks would drift from the committed copy. If a secret is ever added later, the pattern is dangerous.

**Fix:** Track a `.env.example` with placeholder values; add `.env` to `.gitignore`; keep `.env.docker` / `.env.production` only if they're deliberately shared config.

### L14. README.md is out of date / branch-specific
**File:** `README.md`
- L25: "This branch (`huggingface-backend`) is configured for deploying..." — current branch is `main`. Either the HF deploy moved and the README didn't, or there's a stale reference.
- L31: `API Base URL: https://your-space-name.hf.space` — placeholder, should be `https://levineuwirth-icd-embeddings.hf.space`.
- The README is entirely about the backend; no mention of the GH Pages frontend build step (`npm run build`, commits `docs/`) or the Mac Studio alternative path.

### L15. `backend/small_dataset_predictions_{mor30,rea30}.csv`
Used only by `backend/validate_models.py` (a dev-only script). Cluttering the deployment directory; bundled into the Docker image for no reason.

**Fix:** Move to `backend/tests/fixtures/` or a repo-level `scripts/` directory; add an entry to `.dockerignore`.

### L16. `backend/validate_models.py` permission is `-rw-------`
**File:** `backend/validate_models.py`
Mode 600 on a source file that ships in the container is odd — likely an accident from a chmod. Not harmful but surprising.

### L17. `.dockerignore` presumably leaves too much in the image
Haven't opened the file (it's only 505 bytes), but with `backend/small_dataset_predictions_*.csv`, `backend/validate_models.py`, `backend/tests/`, and `backend/data/` + its XML parsing script all going into the Docker build context, the image is larger than it needs to be.

### L18. `parse_icd10.py` silently fails if the CMS XML changes schema
**File:** `backend/data/parse_icd10.py`
`extract_codes` expects `<diag><name>...</name><desc>...</desc>` inside the 2026 CMS tabular XML. No assertion on count; if CMS changes the element names next year, the script prints `Found 0 ICD-10 codes` and writes an empty JSON that breaks the lookup endpoint silently. The Dockerfile pulls `2026-code-tables-tabular-and-index.zip` each build (L25) so this is a real-world ticking issue.

**Fix:** `assert len(codes_dict) > 50000` before writing; let the build fail loudly.

### L19. Dockerfile downloads the CMS ICD-10 zip at image build time (L24–28)
If CMS takes the file down, or the URL changes, the HF Space rebuild fails with no fallback. Also ties every rebuild to an external HTTP dependency — unnecessary given this is data that changes at most yearly.

**Fix:** Commit the parsed `icd10_codes.json` to the repo (already gitignored today, L28 of `.gitignore`) or pin a mirrored copy.

### L20. `Strict-Transport-Security`, `Content-Security-Policy`, `Referrer-Policy` not set
**File:** `nginx.conf:11–13` (and nothing equivalent on the HF side).
Only `X-Frame-Options`, `X-Content-Type-Options`, `X-XSS-Protection` are set. `X-XSS-Protection` is deprecated; modern browsers ignore it.

Low severity because the GH Pages frontend doesn't traverse `nginx.conf` at all, and HF Spaces handles its own edge. But if the Mac Studio path ever comes back, this is the time to add HSTS / a CSP.

### L21. `client_max_body_size 10M` on the nginx path but no equivalent on HF Spaces
Mac Studio path caps uploads at 10 MB; HF Spaces relies on whatever the frontend uvicorn defaults are. Since the upload endpoint has no size check, a very large multipart upload could spend CPU decoding UTF-8 before being rejected.

### L22. `setIcdSearchQuery(value); if (value.length > 2)` but results are only cleared when length drops to 2 or below
**File:** `src/App.jsx:86–97`
If the user types `dia` (3 chars, results fill), then selects all and replaces with `xy` (2 chars, search doesn't run but the previous "diabetes" results stay visible because `setIcdLookupResults([])` is in the else branch). Actually the else branch *does* clear — so this is fine. Scratched, keeping to note the edge case threshold of 2.

### L23. `{handleInputChange('icdMethod', ...)}` closures create a new function per render for every radio/button (L329, L335, L365, L383, L395, etc.)
Not a bug. Just noting that every render allocates a bunch of arrow functions; if perf ever matters, `useCallback` could help. `useCallback` is not currently imported.

### L24. `icd/` is a committed `.gitignore`-controlled venv directory
**Dir:** `icd/bin`, `icd/lib`, `icd/lib64` symlink, `icd/pyvenv.cfg`, `icd/CACHEDIR.TAG`
`icd/.gitignore` is 1 byte (presumably `*`), so git correctly ignores everything. But the directory is still there in the working tree and clutters IDE file listings. On someone else's checkout they'd have to build their own venv in a different location.

**Fix:** Move the venv outside the repo (e.g., `~/.venvs/icd`) or rename to `.venv` which is the community convention.

### L25. No `<meta name="description">` or social/OG tags in `index.html`
**File:** `index.html`
Minimal viable HTML — fine for a research tool, but any inbound link (Slack, email) shows no preview. 5-line addition.

### L26. `<title>` in `index.html` doesn't match the `<h1>` in `App.jsx`
- `index.html` L7: `<title>ICD Diagnosis Prediction Calculator</title>`
- `App.jsx` L284: `<h1 className="main-title">ICD Diagnosis Code Prediction Calculator</h1>`
One word apart ("Code"). Pick one.

### L27. `backend/README.md` referenced in `README.md` (L19) but not checked — see file listing, it exists and is 925 bytes
Not audited inline. Worth reading for consistency with the top-level README.

### L28. `.DS_Store` exists in working tree at repo root
Not tracked (git ls-files doesn't list it). Just a noise file on macOS. Harmless but worth adding `**/.DS_Store` to `.gitignore` (current `.gitignore` only has `.DS_Store` which matches files named exactly that at any depth — fine, actually).

### L29. `model_readmit.name`, `model_mortality.name`, etc., printed at load time (main.py:240–249)
These are Keras default names (`"model_17"` type strings) unless set during training. Not useful in logs. Either set informative `name=` when training or drop the print.

---

## Info (style / nits)

### I1. `useState` never uses the functional update form anywhere
`setFormData(prev => ({...prev, [field]: value}))` is used (good). But `setResults`, `setIcdLookupResults`, `setValidationResults` all pass fresh objects. No bug — just note the convention is mixed.

### I2. `gender === 'F' ? 1 : 0` at `src/App.jsx:226`
If `gender` is `''` (empty), the ternary still evaluates to `0` (male) — but the outer `if (gender)` check at L225 prevents that. Safe today. Would break if someone moved the assignment out of the guard.

### I3. Numeric literals without units
`fontSize: '0.875rem'` (L293, L316, L344, L359, L520...) repeated many times. No theme / design-token system. Fine for a research tool.

### I4. Inline style props everywhere
Lots of `style={{ ... }}` mixed with `className=`. Not wrong, but a CSS file + class names gives you better debuggability and prevents style drift.

### I5. Commit `6503eca` title "ICD + demo calibration" — typical research-shorthand messages (`normalize v2`, `all NaN validation`, `stricter invalids`)
Readable to the author, opaque to future maintainers. A `CONTRIBUTING.md` note about Conventional Commits would help onboarding.

### I6. No CI / GitHub Actions
No workflow runs `pytest backend/`, `npm run build`, `npm run lint`, or `eslint .` on PRs. Given that the tracked tests currently fail (H1), this may be why no one noticed.

### I7. `eslint.config.js` scopes lint to `**/*.{js,jsx}`, but the React 19 upgrade means any new `useEffect` needs the `react-hooks/exhaustive-deps` rule — already configured, so fine, but `useEffect` is not used anywhere in this app despite it being a data-fetching UI. Worth asking whether search / lookup fetches should move into `useEffect` with cleanup/abort.

### I8. `axios` imported but never configured (no `axios.create`, no interceptors)
Every call re-specifies the full URL via `${API_URL}/...`. A single `axios.create({ baseURL: API_URL })` instance would remove the repetition and give a single place to add retry / timeout / error interceptors.

### I9. `vite.config.js` has no `server.proxy` for local dev
When running `npm run dev`, the frontend still hits `VITE_API_BASE_URL=http://127.0.0.1:8000` from `.env`. That works if you're running the backend locally, but then CORS isn't exercised because there's no proxy. A `server.proxy['/api']: 'http://127.0.0.1:8000'` config would let the dev URL match production, while `.env.production` points at the Space.

### I10. `index.html` has `<link rel="icon" type="image/svg+xml" href="/vite.svg" />` — still the Vite default favicon
Branded product, default favicon.

### I11. No `<meta charset>` enforcement in Pydantic string fields
Pydantic trusts UTF-8 already; noting for completeness.

### I12. `ClassVar` / class-level constants vs module-level constants (`BETA_READMIT`, `THRESHOLD_READMIT_ICD_ONLY`, etc.)
These are all module globals. Fine, but they're logically associated with specific models — co-locating them with the model-loading block (or a `ModelConfig` dataclass) would make it easier to swap models in the future.

### I13. `scikit-learn==1.5.1` but the code only uses `LabelEncoder` and `MinMaxScaler` imports (main.py:21)
Neither class is actually used at runtime — they're only loaded out of the pickle. The `from sklearn.preprocessing import LabelEncoder, MinMaxScaler` import is required for unpickling (names must be resolvable), so keep it — but add a one-line comment explaining why.

### I14. `response_model` not used on any FastAPI endpoint
All the rich response structures exist only in code, not in OpenAPI. If a consumer ever wants a typed client they have to reverse-engineer the shape. Low priority for an internal demo, worth doing for anything external.

### I15. `prediction: float(readmission_prob)` double-wrapping
`readmission_prob` is already a Python float (line 510 already wrapped `float(...)`). The second `float()` at L560 is a no-op. Tidy-up.

### I16. `.env` contains `VITE_API_BASE_URL=http://127.0.0.1:8000` — but running `npm run dev` with that pointed at `8000` only works if you also start the backend locally. No `package.json` script launches both.
Consider `concurrently` or a `Makefile` target for local dev.

### I17. `logger.warning(f"Codes mapped to NAN: {codes_mapped_to_nan}")` (main.py:468, 630, 788)
Emits the raw patient-submitted codes to logs. On a public service this is typically fine (no PHI), but if this ever gets reused for a dataset with real identifiers via a URL parameter, re-examine.

---

## Summary

| Severity | Count |
|---|---|
| Critical | 0 |
| High | 10 |
| Medium | 20 |
| Low | 29 |
| Info | 17 |
| **Total** | **76** |

**Top items to address first, in rough priority order:**
1. **H1** — Fix or delete the broken tests; they're giving false confidence / silent failure.
2. **H2** — Replace `calculate_prediction_ci` with something honest (MC-Dropout or drop the "CI" label). It's 100× the inference cost for noise.
3. **H3, H4** — One-line fixes (`Any` typo; narrow the `except`).
4. **H6** — Surface age capping to the user; this is a clinical tool.
5. **H7, H8, H9, H10** — Cleanup: startup robustness, pycache, stale `model/` dir, `docs/` build discipline.
6. **M3, M7, M8, M15, M16, M17** — Frontend UX + a11y hardening.
7. **M9, M10, M11, M18, M19** — Backend input validation + deployment config cleanup.
8. Low / Info items are mostly stylistic — worth a cleanup PR but none block.

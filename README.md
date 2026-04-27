# ICD-10-CM Outcome Prediction

A permutation-invariant deep-learning model for 30-day unplanned readmission and 30-day postdischarge mortality, trained on ICD-10-CM diagnosis-code sets from the Nationwide Readmissions Database. Code, deployed calculator, and supporting infrastructure for the manuscript intended for publication in *JAMA Network Open*.

> Shu L\*, Neuwirth L\*, Wang X\*, Zheng H\*. *Beyond Comorbidity Indices: An Order-Invariant ICD-10-CM Embedding for Readmission and Mortality Prediction.* 

## Why this exists

Most claims-based risk adjustment for short-term clinical outcomes still relies on the Charlson and Elixhauser comorbidity indices, which collapse the patient's full diagnostic picture into a small set of weighted conditions. That collapse is interpretable and widely deployed, but it inevitably discards granularity and may miss clinically meaningful comorbidity patterns and interactions. Recent ML approaches that use the full ICD code set typically simplify or truncate codes, depend on diagnosis ordering (which is administrative, not clinical), or are trained at single sites where coding practices don't generalize.

This project addresses those limitations directly. We embed each ICD-10-CM code as a learned dense vector and aggregate per-discharge code sets through a permutation-invariant Deep Sets operator [Zaheer et al., 2017], producing a single representation that is independent of code ordering. Demographics and socioeconomic context (age, sex, primary payer, ZIP-income quartile) are processed in a parallel tower and fused with the diagnosis representation before the prediction head. The model is trained on **80M+** adult discharges (NRD 2016–2020) and temporally validated on a held-out cohort drawn from **33M+** later discharges (NRD 2021–2022).

## Headline results

Evaluated on a stratified subsample of **3,226,831** temporally held-out discharges:

| Outcome                              | This model (AUROC) | Best comorbidity-index baseline (AUROC) |
|--------------------------------------|--------------------|------------------------------------------|
| 30-day unplanned readmission         | **0.750** (95% CI 0.749–0.750) | 0.655 (CCI)                  |
| 30-day postdischarge in-hospital mortality | **0.856** (95% CI 0.853–0.858) | 0.784 (age-adjusted CCI)     |

DeLong tests for correlated ROC curves: P < .001 for all pairwise comparisons against CCI, age-adjusted CCI, and ECI. F₂ scores at the validation-selected operating threshold: 0.485 vs 0.407 for readmission; 0.053 vs 0.048 for mortality.

Code-level contributions are surfaced via Integrated Gradients [Sundararajan et al., 2017]; ranked attributions are restricted to ICD codes with ≥50 occurrences in the test subsample to suppress instability from rare codes.

## Live calculator

A public, read-only calculator is hosted at:

**[levineuwirth.github.io/icd_embeddings](https://levineuwirth.github.io/icd_embeddings/)**

Accepts a discharge diagnosis list and patient covariates; returns 30-day readmission and postdischarge-mortality probability estimates with per-code Integrated-Gradients attributions. Inputs are not stored. The tool is intended for research and demonstration purposes, **not for clinical decision-making**.

## Repository layout

This is a monorepo:

```text
backend/    FastAPI service (Python / TensorFlow / Keras)
            — model serving, ICD search, file upload, code parsing
src/        React frontend (calculator UI)
model/      Trained model artifacts and ICD-10-CM lookup tables
```

The `huggingface-backend` branch is configured for deploying the backend to Hugging Face Spaces via Docker on port 7860; `main` is the canonical research branch.

## Running locally

**Backend.** From `backend/`:

```bash
uvicorn main:app --reload
```

**Frontend.** From the project root:

```bash
npm install
npm run dev
```

CORS is permissive (`allow_origins=["*"]`) so the frontend can hit any backend deployment during development.

## API endpoints

The backend exposes a small surface for both the frontend and external research use:

| Method | Endpoint                | Purpose                                                          |
|--------|-------------------------|------------------------------------------------------------------|
| GET    | `/`                     | Welcome message                                                  |
| POST   | `/predict/`             | Risk prediction with full patient covariates                     |
| POST   | `/predict_flex/`        | Risk prediction; falls back to ICD-only model if demographics incomplete |
| GET    | `/search_icd/?q=&limit=`| ICD-10-CM code search                                            |
| POST   | `/parse_icd_codes/`     | Parse and validate codes from free text                          |
| POST   | `/upload_icd_file/`     | Upload a file of codes for batch parsing                         |

Example:

```bash
curl -X POST "https://levineuwirth-icd-embeddings.hf.space/predict_flex/" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 65,
    "female": 0,
    "pay1": 1,
    "zipinc_qrtl": 3,
    "icd_codes": ["E11.9", "I10", "J44.0"]
  }'
```

## Data and ethics

The Healthcare Cost and Utilization Project (HCUP) Nationwide Readmissions Database is governed by the HCUP data use agreement. Because the NRD contains de-identified data, the institutional review board determined the study was not human-participants research and that informed consent was not required.

## Authors

- **Liqi Shu** — Department of Neurology, Warren Alpert Medical School, Brown University
- **Levi Neuwirth** — Department of Computer Science, Brown University
- **Xilin Wang** — Department of Mathematics, Brown University
- **Henry Zheng** — Department of Computer Science, Northeastern University

\* Equal-contribution undergraduate authors.

## Citation

```bibtex
@article{shu2026icd,
  author  = {Shu, Liqi and Neuwirth, Levi and Wang, Xilin and Zheng, Henry},
  title   = {Beyond Comorbidity Indices: An Order-Invariant {ICD-10-CM} Embedding for Readmission and Mortality Prediction},
  journal = {JAMA Network Open},
  year    = {2026},
  note    = {Under review.},
  url     = {https://levineuwirth.org/essays/beyond-comorbidity-indices/}
}
```

The full preprint is available at [levineuwirth.org/essays/beyond-comorbidity-indices](https://levineuwirth.org/essays/beyond-comorbidity-indices/).

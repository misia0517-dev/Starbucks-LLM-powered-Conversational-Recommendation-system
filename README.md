# Starbucks Conversational Recommendation System (Stage 1–3)

This project builds an end-to-end recommendation pipeline for Starbucks-style natural language queries:

- **Stage 1 — Constraint Extraction (LLM)**: parse a user query into structured constraints (category, temperature, sugar, etc.)
- **Stage 2 — Candidate Filtering**: apply constraints to the product catalog to narrow to valid candidates
- **Stage 3 — Relevance Ranking**: rank candidates using embeddings + additional signals (and optionally learning-to-rank)

Final output format matches the required submission CSV:
`query_id,products` where `products` is a semicolon-separated ranked list of product IDs.

---

## Repository Contents

### Data
- `products.csv` — full product catalog (attributes: calories, sugar, dairy, caffeine, etc.)
- `queries_train.csv` — training queries with ground-truth ranked answers
- `queries_test.csv` — test queries (text only)

### Code (main scripts)
- `starbucks_stage1_train_export.py`  
  Uses an **LLM prompt** to convert each `query_text` into a structured JSON-like schema, then exports predictions to Excel for validation and error analysis.

- `starbucks_pipeline_train_validate.py`  
  Runs **Stage 2 + Stage 3** on the training set:
  - filters candidates using Stage 1 outputs  
  - computes embedding similarity (Sentence-Transformers) + additional signals  
  - trains/evaluates ranking quality (e.g., NDCG@10) and produces reports

- `starbucks_run_test_submission.py`  
  Applies the same **Stage 2 + Stage 3 ranking pipeline** to the test set and generates the final `submission.csv` in the required format.

### Notebook (optional)
- `starbucks_llm_prompt.ipynb` — notebook used to iterate on prompt design / quick testing

---

## Stage 1 — Constraint Extraction (LLM)

**Goal:** Convert a customer query into a JSON object with EXACT keys:

```json
{
  "category": null,
  "temperature": null,
  "max_calories": null,
  "max_sugar": null,
  "max_price": null,
  "dairy_free": null,
  "vegan": null,
  "caffeine_level": null
}
```

### Key design choices

Strict schema + “JSON only” output to reduce parsing errors
Conservative extraction (use null if not explicitly stated)
Avoid false triggers (e.g., weather “hot day” ≠ “hot drink”)

Run Stage 1 on training queries
python starbucks_stage1_train_export.py

### Output

queries_train_with_stage1_preds.xlsx
predictions sheet: original rows + pred_* columns
accuracy sheet: field-level accuracy vs train constraints
mismatches sheet: examples to improve prompt rules

## Stage 2 — Candidate Filtering
Using the Stage 1 predictions (pred_* columns), we filter the product table:
category/temperature match (when provided)
numeric thresholds: calories <= max_calories, sugar_g <= max_sugar, price <= max_price
dietary constraints:
dairy_free = true → contains_dairy = false
vegan = true → is_vegan = true
caffeine_level mapped from caffeine_mg buckets
This step typically reduces the candidate list to a small set (e.g., 5–20 products).

## Stage 3 — Relevance Ranking
After filtering, we rank remaining candidates using a combination of:
1) Embedding Similarity (semantic)
Embedding model: Sentence-Transformers / all-MiniLM-L6-v2
Create embeddings for:
the query text
each product text (name + description + attributes)
Similarity: cosine similarity
implemented as dot product when vectors are normalized

2) Additional Signals (tie-breakers / refinements)
keyword overlap score
attribute match flags (e.g., category_match, temperature_match)
slack features (distance below thresholds like max_sugar − sugar_g)
3) Optional Learning-to-Rank


Train a LightGBM LambdaMART ranker on (query, product) pairs to learn how to weight the above signals and improve ordering quality.


## Run training evaluation
python starbucks_pipeline_train_validate.py

Outputs include evaluation summaries (e.g., NDCG@10, Recall@10) and error analysis reports.
Generate Test Submission
Once the pipeline is validated on train, run the same logic on the test queries:
python starbucks_run_test_submission.py

### Output
submission.csv with exactly 2 columns:
query_id,products
TEST_001,BEV_010;BEV_011;BEV_001
TEST_002,BEV_007;BEV_005

Order matters: highest-ranked recommendation first.

How to Run (Environment)
Example setup:
conda create -n starbucks python=3.10 -y
conda activate starbucks
pip install -U pip
pip install pandas numpy openpyxl scikit-learn sentence-transformers lightgbm


Notes / Optimization
Product embeddings can be cached to avoid recomputing every run
Candidate filtering dramatically reduces ranking latency per query
Prompt mismatch sheet is used to iteratively improve Stage 1 extraction quality

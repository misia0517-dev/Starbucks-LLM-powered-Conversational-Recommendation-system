import pandas as pd
import json
import re
import subprocess
from typing import Dict, Any




# -----------------------------
# 1) Prompt
# -----------------------------
PROMPT_TEMPLATE = """You are a Starbucks menu query parser.

Your task is to convert a customer query into a JSON object with EXACTLY the following keys and no others:

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

Valid values:
- category: "brewed", "cold_brew", "espresso", "frappuccino", "refresher", "tea"
- temperature: "hot", "iced", "blended"
- caffeine_level: "none", "low", "medium", "high"
- max_calories, max_sugar, max_price: numbers
- dairy_free, vegan: true / false

General rules:
- Use null if a constraint is not explicitly stated.
- Be conservative. Do NOT guess or infer constraints that are not clearly requested.
- Output JSON only. Do NOT include markdown, comments, or explanations.

--------------------
CATEGORY RULES
--------------------
- Set category ONLY if it is explicitly mentioned or clearly implied by drink names.
Examples:
- "latte", "americano", "espresso" → "espresso"
- "black coffee", "drip", "house coffee", "brewed coffee" → "brewed"
- "cold brew" → "cold_brew"
- "tea", "chai", "matcha", "green tea", "herbal tea" → "tea"
- "frappuccino", "frap" → "frappuccino"
- "refresher", "refreshers" → "refresher"
- Match keywords as whole words only. Do NOT match substrings.

--------------------
TEMPERATURE RULES (IMPORTANT)
--------------------
- Only set temperature when the customer is clearly requesting the temperature of the beverage.
- Do NOT set temperature based on weather, season, or environment context.

Examples of CONTEXT-ONLY (temperature must be null):
- "it's hot outside"
- "hot day", "summer heat"
- "it's cold today", "freezing", "chilly morning"
- "the weather is hot/cold"

Only set temperature in these cases:

Hot:
- "hot coffee", "hot tea", "hot latte"
- "served hot", "extra hot"
- "something warm"

Iced:
- "iced", "on ice", "over ice"
- "iced coffee", "iced latte"

Blended:
- "blended", "frozen", "slush", "slushy"
- "frappuccino", "frap"

Additional rules:
- Do NOT infer temperature from category names (e.g., do NOT infer iced from "cold brew").
- If both context and explicit drink temperature appear, follow the explicit drink temperature.

--------------------
NUTRITION RULES
--------------------
- Set max_sugar ONLY if a sugar amount is explicitly mentioned.
  Example: "under 45 grams of sugar"
- Set max_calories ONLY if calories are explicitly mentioned.
- Do NOT infer from words like "sweet", "healthy", "light".

--------------------
PRICE RULES
--------------------
- Set max_price ONLY if money is explicitly mentioned.
  Examples:
  - "$5"
  - "under 6 dollars"
  - "max $4.5"
- Do NOT treat numbers related to sugar or calories as price.

--------------------
DAIRY / VEGAN RULES
--------------------
- Set dairy_free = true ONLY if the user explicitly asks for:
  "no dairy", "dairy-free", "lactose-free", "no milk", "non-dairy"
- Set vegan = true ONLY if the user explicitly says:
  "vegan", "plant-based", "no animal products"
- Do NOT assume vegan implies dairy_free.

--------------------
CAFFEINE RULES
--------------------
- "no caffeine", "decaf", "without caffeine" → "none"
- "not too much caffeine", "mild", "light caffeine" → "low"
- "regular caffeine" → "medium"
- "need caffeine", "pick me up", "extra shot", "strong coffee", "high caffeine" → "high"
- If caffeine level is not clearly specified, use null.

--------------------
Customer query:
"{query_text}"
"""


EXPECTED_KEYS = [
    "category", "temperature", "max_calories", "max_sugar",
    "max_price", "dairy_free", "vegan", "caffeine_level"
]


# -----------------------------
# 2) Helpers
# -----------------------------
def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1]
        s = s.replace("json", "", 1).strip()
    return s.strip()


def normalize_pred(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: d.get(k, None) for k in EXPECTED_KEYS}

    # numeric
    for nk in ["max_calories", "max_sugar", "max_price"]:
        if out[nk] is not None:
            try:
                out[nk] = float(out[nk])
            except Exception:
                out[nk] = None

    # boolean
    for bk in ["dairy_free", "vegan"]:
        if out[bk] is not None and not isinstance(out[bk], bool):
            v = str(out[bk]).lower()
            if v in ["true", "t", "yes", "1"]:
                out[bk] = True
            elif v in ["false", "f", "no", "0"]:
                out[bk] = False
            else:
                out[bk] = None

    return out


def eq(a, b) -> bool:
    if pd.isna(a): a = None
    if pd.isna(b): b = None
    if a is None and b is None:
        return True
    if (a is None) ^ (b is None):
        return False
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except Exception:
            return False
    return a == b


# -----------------------------
# 3) Backends
# -----------------------------
def llm_call_ollama(query_text: str, model: str) -> Dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(query_text=query_text.replace('"', '\\"'))
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
        check=True
    )
    raw = strip_code_fences(result.stdout)
    d = json.loads(raw)
    return normalize_pred(d)


def llm_call_openai(query_text: str, model: str) -> Dict[str, Any]:
    # Requires: pip install openai  + export OPENAI_API_KEY=...
    from openai import OpenAI
    client = OpenAI()
    prompt = PROMPT_TEMPLATE.format(query_text=query_text.replace('"', '\\"'))

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )
    raw = strip_code_fences(resp.choices[0].message.content)
    d = json.loads(raw)
    return normalize_pred(d)


def llm_call_rules(query_text: str) -> Dict[str, Any]:
    # Fallback (not an LLM call) — use only if you cannot run openai/ollama.
    q = query_text.lower()
    out = {k: None for k in EXPECTED_KEYS}

    def has_word(w): 
        return re.search(rf"\b{re.escape(w)}\b", q) is not None

    # category
    if "cold brew" in q or "coldbrew" in q:
        out["category"] = "cold_brew"
    elif any(has_word(w) for w in ["latte","americano","espresso","cappuccino","macchiato","mocha","shot"]):
        out["category"] = "espresso"
    elif any(has_word(w) for w in ["frappuccino","frap"]):
        out["category"] = "frappuccino"
    elif any(has_word(w) for w in ["refresher","refreshers"]):
        out["category"] = "refresher"
    elif any(has_word(w) for w in ["tea","chai","matcha"]):
        out["category"] = "tea"
    elif any(phrase in q for phrase in ["black coffee","drip","house coffee","brewed coffee","brewed"]):
        out["category"] = "brewed"

    # temperature explicit
    if any(has_word(w) for w in ["steaming","hot"]):
        out["temperature"] = "hot"
    if has_word("iced") or "on ice" in q:
        out["temperature"] = "iced"
    if any(has_word(w) for w in ["blended","frozen","slush","frap"]):
        out["temperature"] = "blended"

    # sugar / calories
    m = re.search(r"(under|less than|<=|at most|max(?:imum)?|no more than)\s*(\d+(?:\.\d+)?)\s*(g|grams)?\s*(of\s*)?sugar", q)
    if m: out["max_sugar"] = float(m.group(2))

    m = re.search(r"(under|less than|<=|at most|max(?:imum)?|no more than)\s*(\d+(?:\.\d+)?)\s*(cal|cals|calories)\b", q)
    if m: out["max_calories"] = float(m.group(2))

    # price (only money context)
    m = re.search(r"(under|less than|<=|at most|max(?:imum)?|no more than)\s*\$?\s*(\d+(?:\.\d+)?)", q)
    if m and ("$" in q or "dollar" in q or "bucks" in q or "price" in q or "budget" in q or "cheaper" in q):
        out["max_price"] = float(m.group(2))

    # dairy / vegan explicit
    if re.search(r"(dairy[-\s]?free|no dairy|avoid dairy|lactose[-\s]?free|no milk|non[-\s]?dairy)", q):
        out["dairy_free"] = True
    if re.search(r"(vegan|plant[-\s]?based|no animal|no animal products)", q):
        out["vegan"] = True

    # caffeine
    if re.search(r"(no caffeine|without caffeine|decaf|caffeine[-\s]?free)", q):
        out["caffeine_level"] = "none"
    elif re.search(r"(not too much caffeine|mild|light caffeine)", q):
        out["caffeine_level"] = "low"
    elif re.search(r"(regular caffeine)", q):
        out["caffeine_level"] = "medium"
    elif re.search(r"(need caffeine|pick me up|extra shot|strong|high caffeine)", q):
        out["caffeine_level"] = "high"

    return out


def stage1_extract(query_text: str, backend: str, model: str) -> Dict[str, Any]:
    if backend == "ollama":
        return llm_call_ollama(query_text, model=model)
    if backend == "openai":
        return llm_call_openai(query_text, model=model)
    if backend == "rules":
        return llm_call_rules(query_text)
    raise ValueError(f"Unknown backend: {backend}")


# -----------------------------
# 4) Main
# -----------------------------
def main():
    # ====== EDIT THESE SETTINGS ======
    TRAIN_CSV = "queries_train.csv"
    OUT_XLSX = "queries_train_with_stage1_preds.xlsx"
    BACKEND = "openai"      # "ollama" / "openai" / "rules"
    MODEL = "gpt-4.1-mini"      # openai example: "gpt-4.1-mini"
    LIMIT = 0               # 0 = all
    # ================================

    df = pd.read_csv(TRAIN_CSV)
    if LIMIT and LIMIT > 0:
        df = df.head(LIMIT).copy()

    preds = []
    for i, qt in enumerate(df["query_text"].tolist(), start=1):
        try:
            d = stage1_extract(qt, backend=BACKEND, model=MODEL)
            d["_error"] = None
        except Exception as e:
            d = {k: None for k in EXPECTED_KEYS}
            d["_error"] = str(e)
        preds.append(d)

        if i % 10 == 0:
            print(f"Processed {i}/{len(df)}")

    pred_df = pd.DataFrame(preds)

    # attach preds back
    out = df.copy()
    for k in EXPECTED_KEYS:
        out[f"pred_{k}"] = pred_df[k]
    if "_error" in pred_df.columns:
        out["_error"] = pred_df["_error"]

    # compare with ground-truth constraint_* columns (if present)
    gt_map = {
        "category": "constraint_category",
        "temperature": "constraint_temperature",
        "max_calories": "constraint_max_calories",
        "max_sugar": "constraint_max_sugar",
        "max_price": "constraint_max_price",
        "dairy_free": "constraint_dairy_free",
        "vegan": "constraint_vegan",
        "caffeine_level": "constraint_caffeine_level",
    }

    acc_rows = []
    mismatches = []

    for k, gt_col in gt_map.items():
        if gt_col not in out.columns:
            continue

        pred_col = f"pred_{k}"
        correct = [eq(p, g) for p, g in zip(out[pred_col], out[gt_col])]
        acc = sum(correct) / len(correct) if len(correct) else 0.0
        acc_rows.append({"field": k, "accuracy": acc})

        for idx, ok in enumerate(correct):
            if not ok:
                mismatches.append({
                    "query_id": out.iloc[idx].get("query_id"),
                    "query_text": out.iloc[idx].get("query_text"),
                    "field": k,
                    "pred": out.iloc[idx][pred_col],
                    "gt": out.iloc[idx][gt_col],
                })

    acc_df = pd.DataFrame(acc_rows).sort_values("accuracy", ascending=True)
    mism_df = pd.DataFrame(mismatches)

    # export to Excel with multiple sheets
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="predictions", index=False)
        acc_df.to_excel(writer, sheet_name="accuracy", index=False)
        mism_df.to_excel(writer, sheet_name="mismatches", index=False)

    print(f"\nSaved Excel: {OUT_XLSX}")
    print("Sheets: predictions, accuracy, mismatches")
    print(f"Backend: {BACKEND}, Model: {MODEL}")


if __name__ == "__main__":
    main()

import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PRODUCTS_CSV = "products.csv"
TEST_STAGE1_XLSX = "queries_test_with_stage1_preds.xlsx"
TEST_STAGE1_SHEET = "predictions"
OUT_SUBMISSION_CSV = "submission.csv"

HYBRID = True
W_EMB = 1.0
W_KEYWORD = 0.2

EMB_MODEL_NAME = "all-MiniLM-L6-v2"
EMB_CACHE_PATH = "product_embeddings.npy"
TEXT_CACHE_PATH = "product_texts.csv"


MAX_RETURN = None


def build_product_text(df: pd.DataFrame) -> pd.Series:
    def row_to_text(r):
        parts = [
            f"name: {r.get('name','')}",
            f"description: {r.get('description','')}",
            f"category: {r.get('category','')}",
            f"temperature: {r.get('temperature','')}",
            f"calories: {r.get('calories','')}",
            f"sugar_g: {r.get('sugar_g','')}",
            f"price: {r.get('price','')}",
            f"caffeine_mg: {r.get('caffeine_mg','')}",
        ]
        if "contains_dairy" in r:
            parts.append("contains_dairy" if r["contains_dairy"] else "no_dairy")
        if "is_vegan" in r:
            parts.append("vegan" if r["is_vegan"] else "not_vegan")
        return " | ".join(map(str, parts))
    return df.apply(row_to_text, axis=1)


def keyword_score(query: str, text: str) -> float:
    q = str(query).lower()
    t = str(text).lower()
    q_tokens = set(re.findall(r"[a-z]+", q))
    t_tokens = set(re.findall(r"[a-z]+", t))
    overlap = len(q_tokens & t_tokens)

    bonus = 0
    for phrase in ["cold brew","latte","matcha","refresher","frappuccino","americano","espresso","chai","green tea","tea"]:
        if phrase in q and phrase in t:
            bonus += 3
    return overlap + bonus


# ---------- stage2 filter ----------
def filter_products(products: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    df = products.copy()

    # category
    if pd.notna(row.get("pred_category", np.nan)):
        df = df[df["category"] == row["pred_category"]]

    # temperature
    if pd.notna(row.get("pred_temperature", np.nan)):
        df = df[df["temperature"] == row["pred_temperature"]]

    # max calories
    if pd.notna(row.get("pred_max_calories", np.nan)):
        df = df[df["calories"] <= float(row["pred_max_calories"])]

    # max sugar
    if pd.notna(row.get("pred_max_sugar", np.nan)):
        df = df[df["sugar_g"] <= float(row["pred_max_sugar"])]

    # max price
    if pd.notna(row.get("pred_max_price", np.nan)):
        df = df[df["price"] <= float(row["pred_max_price"])]

    # dairy_free
    if row.get("pred_dairy_free", None) is True and "contains_dairy" in df.columns:
        df = df[df["contains_dairy"] == False]

    # vegan
    if row.get("pred_vegan", None) is True and "is_vegan" in df.columns:
        df = df[df["is_vegan"] == True]

    # caffeine_level
    lvl = row.get("pred_caffeine_level", None)
    if pd.notna(lvl) and "caffeine_mg" in df.columns:
        if lvl == "none":
            df = df[df["caffeine_mg"] == 0]
        elif lvl == "low":
            df = df[(df["caffeine_mg"] > 0) & (df["caffeine_mg"] <= 80)]
        elif lvl == "medium":
            df = df[(df["caffeine_mg"] >= 80) & (df["caffeine_mg"] <= 160)]
        elif lvl == "high":
            df = df[df["caffeine_mg"] > 160]

    return df.copy()


# ---------- embeddings cache ----------
def ensure_embeddings(products: pd.DataFrame, model: SentenceTransformer):
    if not os.path.exists(TEXT_CACHE_PATH):
        products = products.copy()
        products["product_text"] = build_product_text(products)
        products[["product_id", "product_text"]].to_csv(TEXT_CACHE_PATH, index=False)
    else:
        txt = pd.read_csv(TEXT_CACHE_PATH)
        products = products.merge(txt, on="product_id", how="left")

    if not os.path.exists(EMB_CACHE_PATH):
        emb = model.encode(products["product_text"].tolist(), normalize_embeddings=True)
        np.save(EMB_CACHE_PATH, emb)
    else:
        emb = np.load(EMB_CACHE_PATH)

    id_to_idx = {pid: i for i, pid in enumerate(products["product_id"].tolist())}
    return products, emb, id_to_idx


# ---------- stage3 ranking ----------
def rank_candidates(query_text: str, cand_df: pd.DataFrame, products_all: pd.DataFrame,
                    prod_emb: np.ndarray, id_to_idx: dict, model: SentenceTransformer) -> list:
    if cand_df.empty:
        return []

    # embedding score
    q_emb = model.encode([query_text], normalize_embeddings=True)[0]
    cand_ids = cand_df["product_id"].tolist()
    idxs = [id_to_idx[pid] for pid in cand_ids]
    emb_scores = (prod_emb[idxs] @ q_emb)

    # keyword score (use product_text from products_all)
    ptext_map = products_all.set_index("product_id")["product_text"].to_dict()

    scores = []
    for pid, e in zip(cand_ids, emb_scores):
        kw = keyword_score(query_text, ptext_map.get(pid, ""))
        s = (W_EMB * float(e)) + (W_KEYWORD * float(kw)) if HYBRID else float(e)
        scores.append(s)

    order = np.argsort(-np.array(scores))
    ranked = [cand_ids[i] for i in order]

    if MAX_RETURN is not None:
        ranked = ranked[:MAX_RETURN]
    return ranked


def main():
    products = pd.read_csv(PRODUCTS_CSV)
    preds = pd.read_excel(TEST_STAGE1_XLSX, sheet_name=TEST_STAGE1_SHEET)

    model = SentenceTransformer(EMB_MODEL_NAME)
    products_all, prod_emb, id_to_idx = ensure_embeddings(products, model)

    out_rows = []
    for _, r in preds.iterrows():
        qid = r["query_id"]
        qtext = r["query_text"]

        cand_df = filter_products(products_all, r)
        ranked_ids = rank_candidates(qtext, cand_df, products_all, prod_emb, id_to_idx, model)

        out_rows.append({
            "query_id": qid,
            "products": ";".join(ranked_ids)
        })

    sub = pd.DataFrame(out_rows)
    sub.to_csv(OUT_SUBMISSION_CSV, index=False)
    print(f"✅ Saved submission file: {OUT_SUBMISSION_CSV}")
    print(sub.head(5))


if __name__ == "__main__":
    main()

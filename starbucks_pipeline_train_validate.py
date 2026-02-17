import os, re, ast, math
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

PRODUCTS_CSV = "products.csv"
TRAIN_CSV = "queries_train.csv"
STAGE1_PREDS_XLSX = "queries_train_with_stage1_preds.xlsx"
STAGE1_PREDS_SHEET = "predictions"

EMB_MODEL_NAME = "all-MiniLM-L6-v2"
EMB_CACHE_PATH = "product_embeddings.npy"
TEXT_CACHE_PATH = "product_texts.csv"

K_EVAL = 10


MAX_CANDS_PER_QUERY = 50  # set None for no cap

# Outputs
OUT_CV_XLSX = "train_5fold_cv_report.xlsx"
OUT_FINAL_MODEL_PATH = "lgbm_ranker_full_train.txt"
OUT_FINAL_TRAIN_PRED_CSV = "train_predictions_full_model.csv"
OUT_FINAL_TRAIN_EVAL_XLSX = "train_eval_report_full_model.xlsx"


# -------------------------
# Column mapping
# -------------------------
PID = "product_id"
PNAME = "name"
PDESC = "description"
PCAT = "category"
PTEMP = "temperature"
PCAL = "calories"
PSUGAR = "sugar_g"
PPRICE = "price"
PDAIRY = "contains_dairy"
PVEGAN = "is_vegan"
PCAFF = "caffeine_mg"

QID = "query_id"
QTEXT = "query_text"
GT = "relevant_products"


# -------------------------
# Metrics
# -------------------------
def parse_gt(val):
    """Parse relevant_products into a list of product_ids."""
    if pd.isna(val) or str(val).strip() == "":
        return []
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            out = ast.literal_eval(s)
            return list(out) if isinstance(out, (list, tuple)) else []
        except Exception:
            return []
    return [x for x in s.split(";") if x.strip()]


def mrr(pred, gt):
    gt_set = set(gt)
    for i, p in enumerate(pred, start=1):
        if p in gt_set:
            return 1.0 / i
    return 0.0


def recall_at_k(pred, gt, k=10):
    gt_set = set(gt)
    if len(gt_set) == 0:
        return np.nan
    return len([p for p in pred[:k] if p in gt_set]) / len(gt_set)


def ndcg_at_k(pred, gt, k=10):
    if not gt:
        return 0.0
    gt_rank = {pid: i + 1 for i, pid in enumerate(gt)}  # 1-based

    def gain(pid):
        if pid not in gt_rank:
            return 0.0
        rel = 1.0 / gt_rank[pid]   # higher if earlier in gt
        return (2 ** rel - 1)

    def dcg(lst):
        s = 0.0
        for i, pid in enumerate(lst[:k], start=1):
            s += gain(pid) / math.log2(i + 1)
        return s

    ideal = gt[:k]
    denom = dcg(ideal)
    return 0.0 if denom == 0 else dcg(pred) / denom


# Product text + keyword
def build_product_text(products: pd.DataFrame) -> pd.Series:
    def row_to_text(r):
        parts = []
        if pd.notna(r.get(PNAME, None)): parts.append(f"Name: {r[PNAME]}")
        if pd.notna(r.get(PDESC, None)) and str(r[PDESC]).strip():
            parts.append(f"Description: {r[PDESC]}")
        if pd.notna(r.get(PCAT, None)): parts.append(f"Category: {r[PCAT]}")
        if pd.notna(r.get(PTEMP, None)): parts.append(f"Served: {r[PTEMP]}")
        if pd.notna(r.get(PCAL, None)): parts.append(f"Calories: {r[PCAL]}")
        if pd.notna(r.get(PSUGAR, None)): parts.append(f"Sugar: {r[PSUGAR]} g")
        if pd.notna(r.get(PPRICE, None)): parts.append(f"Price: {r[PPRICE]}")
        if pd.notna(r.get(PCAFF, None)): parts.append(f"Caffeine: {r[PCAFF]} mg")
        if pd.notna(r.get(PDAIRY, None)): parts.append("Contains dairy" if r[PDAIRY] else "Dairy-free")
        if pd.notna(r.get(PVEGAN, None)): parts.append("Vegan" if r[PVEGAN] else "Not vegan")
        return " | ".join(map(str, parts))
    return products.apply(row_to_text, axis=1)


def keyword_score(query: str, text: str) -> float:
    q = str(query).lower()
    t = str(text).lower()
    q_tokens = set(re.findall(r"[a-z]+", q))
    t_tokens = set(re.findall(r"[a-z]+", t))
    overlap = len(q_tokens & t_tokens)

    bonus = 0
    for phrase in ["cold brew","latte","matcha","refresher","frappuccino","americano","espresso","chai","green tea"]:
        if phrase in q and phrase in t:
            bonus += 3
    return overlap + bonus


# Stage 2 filter (uses Stage1 pred_* columns)
def filter_products(products: pd.DataFrame, r: pd.Series) -> pd.DataFrame:
    df = products

    if pd.notna(r.get("pred_category", np.nan)) and PCAT in df.columns:
        df = df[df[PCAT] == r["pred_category"]]
    if pd.notna(r.get("pred_temperature", np.nan)) and PTEMP in df.columns:
        df = df[df[PTEMP] == r["pred_temperature"]]
    if pd.notna(r.get("pred_max_calories", np.nan)) and PCAL in df.columns:
        df = df[df[PCAL] <= float(r["pred_max_calories"])]
    if pd.notna(r.get("pred_max_sugar", np.nan)) and PSUGAR in df.columns:
        df = df[df[PSUGAR] <= float(r["pred_max_sugar"])]
    if pd.notna(r.get("pred_max_price", np.nan)) and PPRICE in df.columns:
        df = df[df[PPRICE] <= float(r["pred_max_price"])]

    if r.get("pred_dairy_free", None) is True and PDAIRY in df.columns:
        df = df[df[PDAIRY] == False]
    if r.get("pred_vegan", None) is True and PVEGAN in df.columns:
        df = df[df[PVEGAN] == True]

    lvl = r.get("pred_caffeine_level", None)
    if pd.notna(lvl) and PCAFF in df.columns:
        if lvl == "none":
            df = df[df[PCAFF] == 0]
        elif lvl == "low":
            df = df[(df[PCAFF] > 0) & (df[PCAFF] <= 80)]
        elif lvl == "medium":
            df = df[(df[PCAFF] >= 80) & (df[PCAFF] <= 160)]
        elif lvl == "high":
            df = df[df[PCAFF] > 160]

    return df.copy()


# Embedding cache
def ensure_embeddings_cache(products: pd.DataFrame, model: SentenceTransformer):
    # cache product_text
    if not os.path.exists(TEXT_CACHE_PATH):
        products = products.copy()
        products["product_text"] = build_product_text(products)
        products[[PID, "product_text"]].to_csv(TEXT_CACHE_PATH, index=False)
    else:
        txt_cache = pd.read_csv(TEXT_CACHE_PATH)
        products = products.merge(txt_cache, on=PID, how="left")

    # cache embeddings
    if not os.path.exists(EMB_CACHE_PATH):
        print("Embedding cache not found. Computing embeddings for all products...")
        emb = model.encode(products["product_text"].tolist(), normalize_embeddings=True)
        np.save(EMB_CACHE_PATH, emb)
        print("Saved:", EMB_CACHE_PATH)
    else:
        emb = np.load(EMB_CACHE_PATH)

    # mapping from id to embedding index
    id_to_idx = {pid: i for i, pid in enumerate(products[PID].tolist())}
    return products, emb, id_to_idx


# Build ranking dataset
FEATURES = [
    "emb_score", "kw_score",
    "calories", "sugar_g", "caffeine_mg", "price",
    "contains_dairy", "is_vegan",
    "category_match", "temperature_match",
    "sugar_slack", "cal_slack", "price_slack",
]

def make_features(query_text, pred_row, prod_row, prod_text, emb_score):
    kw = keyword_score(query_text, prod_text)

    cal = prod_row.get(PCAL, np.nan)
    sugar = prod_row.get(PSUGAR, np.nan)
    caff = prod_row.get(PCAFF, np.nan)
    price = prod_row.get(PPRICE, np.nan)

    dairy = prod_row.get(PDAIRY, np.nan)
    vegan = prod_row.get(PVEGAN, np.nan)

    cat_match = 1.0 if pd.notna(pred_row.get("pred_category", np.nan)) and prod_row.get(PCAT, None) == pred_row.get("pred_category") else 0.0
    temp_match = 1.0 if pd.notna(pred_row.get("pred_temperature", np.nan)) and prod_row.get(PTEMP, None) == pred_row.get("pred_temperature") else 0.0

    sugar_slack = 0.0
    if pd.notna(pred_row.get("pred_max_sugar", np.nan)) and pd.notna(sugar):
        sugar_slack = float(pred_row["pred_max_sugar"]) - float(sugar)

    cal_slack = 0.0
    if pd.notna(pred_row.get("pred_max_calories", np.nan)) and pd.notna(cal):
        cal_slack = float(pred_row["pred_max_calories"]) - float(cal)

    price_slack = 0.0
    if pd.notna(pred_row.get("pred_max_price", np.nan)) and pd.notna(price):
        price_slack = float(pred_row["pred_max_price"]) - float(price)

    return {
        "emb_score": float(emb_score),
        "kw_score": float(kw),
        "calories": float(cal) if pd.notna(cal) else np.nan,
        "sugar_g": float(sugar) if pd.notna(sugar) else np.nan,
        "caffeine_mg": float(caff) if pd.notna(caff) else np.nan,
        "price": float(price) if pd.notna(price) else np.nan,
        "contains_dairy": float(dairy) if pd.notna(dairy) else np.nan,
        "is_vegan": float(vegan) if pd.notna(vegan) else np.nan,
        "category_match": float(cat_match),
        "temperature_match": float(temp_match),
        "sugar_slack": float(sugar_slack),
        "cal_slack": float(cal_slack),
        "price_slack": float(price_slack),
    }


def make_label(gt_list, pid, cap=10):
    """
    Convert ranked ground-truth list into a capped relevance label.
    Avoids "label mapping" issues and prevents overfitting to very fine ranks.
    """
    if pid not in gt_list:
        return 0
    rank = gt_list.index(pid)  # 0-based
    return max(1, cap - rank)


def build_rank_table(df_queries, products_all, prod_emb, id_to_idx, model):
    """
    Build a (query, product) table with features + labels.
    group_sizes must align with query order for LGBMRanker.
    """
    rows = []
    group_sizes = []

    for _, r in df_queries.iterrows():
        qid = r[QID]
        qtext = r[QTEXT]
        gt_list = r["gt_list"]

        cand = filter_products(products_all, r)
        if MAX_CANDS_PER_QUERY is not None and len(cand) > MAX_CANDS_PER_QUERY:
            # NOTE: simple cap; you can replace with top-N by emb_score to be safer
            cand = cand.head(MAX_CANDS_PER_QUERY).copy()

        cand_ids = cand[PID].tolist()
        group_sizes.append(len(cand_ids))

        if len(cand_ids) == 0:
            continue

        q_emb = model.encode([qtext], normalize_embeddings=True)[0]
        idxs = [id_to_idx[pid] for pid in cand_ids]
        emb_mat = prod_emb[idxs]
        emb_scores = emb_mat @ q_emb  # cosine sim since normalized

        # build one row per candidate
        for pid_i, emb_s in zip(cand_ids, emb_scores):
            prow = cand.loc[cand[PID] == pid_i].iloc[0]
            ptext = products_all.loc[products_all[PID] == pid_i, "product_text"].iloc[0]
            feats = make_features(qtext, r, prow, ptext, emb_s)
            y = make_label(gt_list, pid_i)
            rows.append({
                "query_id": qid,
                "product_id": pid_i,
                "label": y,
                **feats
            })

    df_rank = pd.DataFrame(rows)
    return df_rank, group_sizes


# Train + Predict
def train_ranker(df_rank, group_sizes):
    X = df_rank[FEATURES].copy()
    y = df_rank["label"].astype(float).values

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=20,
        random_state=42,
    )
    ranker.fit(X, y, group=group_sizes)
    return ranker


def rank_with_model(ranker, r, products_all, prod_emb, id_to_idx, model):
    qtext = r[QTEXT]
    cand = filter_products(products_all, r)
    if cand.empty:
        return []

    cand_ids = cand[PID].tolist()
    q_emb = model.encode([qtext], normalize_embeddings=True)[0]
    idxs = [id_to_idx[pid] for pid in cand_ids]
    emb_mat = prod_emb[idxs]
    emb_scores = emb_mat @ q_emb

    feat_rows = []
    for pid_i, emb_s in zip(cand_ids, emb_scores):
        prow = cand.loc[cand[PID] == pid_i].iloc[0]
        ptext = products_all.loc[products_all[PID] == pid_i, "product_text"].iloc[0]
        feat_rows.append(make_features(qtext, r, prow, ptext, emb_s))

    X = pd.DataFrame(feat_rows)[FEATURES]
    scores = ranker.predict(X)
    order = np.argsort(-scores)
    return [cand_ids[i] for i in order]


def evaluate_queries(ranker, df_queries, products_all, prod_emb, id_to_idx, model, k=10):
    per_query = []
    for _, r in df_queries.iterrows():
        qid = r[QID]
        gt_list = r["gt_list"]
        pred = rank_with_model(ranker, r, products_all, prod_emb, id_to_idx, model)
        cand = filter_products(products_all, r)

        per_query.append({
            "query_id": qid,
            "query_text": r[QTEXT],
            "num_candidates": len(cand),
            "gt_len": len(gt_list),
            "mrr": mrr(pred, gt_list),
            f"recall@{k}": recall_at_k(pred, gt_list, k),
            f"ndcg@{k}": ndcg_at_k(pred, gt_list, k),
            "pred_top10": ";".join(pred[:10]),
            "gt_list": ";".join(gt_list),
            "gt_in_candidates": any(x in set(cand[PID].tolist()) for x in gt_list) if len(gt_list) else np.nan
        })

    df = pd.DataFrame(per_query)
    summary = {
        "MRR_mean": df["mrr"].mean(),
        f"Recall@{k}_mean": df[f"recall@{k}"].mean(skipna=True),
        f"NDCG@{k}_mean": df[f"ndcg@{k}"].mean(),
        "Avg_num_candidates": df["num_candidates"].mean(),
        "Queries_with_zero_candidates": int((df["num_candidates"] == 0).sum()),
        "n_queries": len(df),
    }
    return df, summary


# Main
def main():
    # Load data
    products = pd.read_csv(PRODUCTS_CSV)
    train = pd.read_csv(TRAIN_CSV)
    preds = pd.read_excel(STAGE1_PREDS_XLSX, sheet_name=STAGE1_PREDS_SHEET)

    train["gt_list"] = train[GT].apply(parse_gt)
    pred_cols = [c for c in preds.columns if c.startswith("pred_")]

    merged = train[[QID, QTEXT, "gt_list"]].merge(
        preds[[QID, QTEXT] + pred_cols],
        on=[QID, QTEXT],
        how="left"
    )

    # Embedding model + cache
    model = SentenceTransformer(EMB_MODEL_NAME)
    products_all, prod_emb, id_to_idx = ensure_embeddings_cache(products, model)

    # 5-FOLD CV
    gkf = GroupKFold(n_splits=5)

    groups = merged[QID].values
    fold_summaries = []
    fold_details = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(merged, groups=groups), start=1):
        train_fold = merged.iloc[tr_idx].reset_index(drop=True)
        valid_fold = merged.iloc[va_idx].reset_index(drop=True)

        print(f"\nFold {fold}/5: build training rank table...")
        df_rank_tr, group_sizes_tr = build_rank_table(train_fold, products_all, prod_emb, id_to_idx, model)
        print(f"  Train pairs: {len(df_rank_tr)} | Train queries: {len(train_fold)}")

        if len(df_rank_tr) == 0:
            print("  WARNING: No training pairs in this fold. Skipping.")
            continue

        print(f"Fold {fold}/5: train ranker...")
        ranker = train_ranker(df_rank_tr, group_sizes_tr)

        print(f"Fold {fold}/5: evaluate on validation queries...")
        per_q_df, summ = evaluate_queries(ranker, valid_fold, products_all, prod_emb, id_to_idx, model, k=K_EVAL)
        summ["fold"] = fold
        summ["Embedding_model"] = EMB_MODEL_NAME
        summ["Ranker"] = "LightGBM LambdaMART"

        fold_summaries.append(summ)
        per_q_df["fold"] = fold
        fold_details.append(per_q_df)

        print(f"  Fold {fold} summary:",
              f"MRR={summ['MRR_mean']:.4f},",
              f"Recall@{K_EVAL}={summ[f'Recall@{K_EVAL}_mean']:.4f},",
              f"NDCG@{K_EVAL}={summ[f'NDCG@{K_EVAL}_mean']:.4f}")

    cv_summary_df = pd.DataFrame(fold_summaries)
    cv_detail_df = pd.concat(fold_details, ignore_index=True) if fold_details else pd.DataFrame()

    # Save CV report
    with pd.ExcelWriter(OUT_CV_XLSX, engine="openpyxl") as writer:
        cv_summary_df.to_excel(writer, sheet_name="fold_summary", index=False)
        cv_detail_df.to_excel(writer, sheet_name="per_query", index=False)

    print("\n✅ Saved 5-fold CV report:", OUT_CV_XLSX)

    if not cv_summary_df.empty:
        mean_row = {
            "MRR_mean": cv_summary_df["MRR_mean"].mean(),
            f"Recall@{K_EVAL}_mean": cv_summary_df[f"Recall@{K_EVAL}_mean"].mean(),
            f"NDCG@{K_EVAL}_mean": cv_summary_df[f"NDCG@{K_EVAL}_mean"].mean(),
            "Avg_num_candidates": cv_summary_df["Avg_num_candidates"].mean(),
        }
        print("\n5-FOLD CV MEAN:")
        print(pd.DataFrame([mean_row]).to_string(index=False))

    # Train a FINAL model on ALL training queries
    # (use after CV to generate test submissions)
    print("\nTraining FINAL model on ALL training queries...")
    df_rank_all, group_sizes_all = build_rank_table(merged, products_all, prod_emb, id_to_idx, model)
    ranker_full = train_ranker(df_rank_all, group_sizes_all)
    ranker_full.booster_.save_model(OUT_FINAL_MODEL_PATH)
    print("✅ Saved final model:", OUT_FINAL_MODEL_PATH)

    # Evaluate final model on TRAIN (for reference only)
    per_query_df, summary = evaluate_queries(ranker_full, merged, products_all, prod_emb, id_to_idx, model, k=K_EVAL)
    summary_df = pd.DataFrame([{
        **summary,
        "Embedding_model": EMB_MODEL_NAME,
        "Ranker": "LightGBM LambdaMART",
        "Model_path": OUT_FINAL_MODEL_PATH,
        "NOTE": "Train-set evaluation only (use CV for generalization)."
    }])
    worst_20 = per_query_df.sort_values(["mrr", f"ndcg@{K_EVAL}"], ascending=[True, True]).head(20)

    # Save final train predictions CSV (not submission; just for sanity)
    pred_ranked = {}
    for _, r in merged.iterrows():
        pred_ranked[r[QID]] = rank_with_model(ranker_full, r, products_all, prod_emb, id_to_idx, model)

    out_rows = [{"query_id": qid, "products": ";".join(pids)} for qid, pids in pred_ranked.items()]
    pd.DataFrame(out_rows).to_csv(OUT_FINAL_TRAIN_PRED_CSV, index=False)

    with pd.ExcelWriter(OUT_FINAL_TRAIN_EVAL_XLSX, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        per_query_df.to_excel(writer, sheet_name="per_query", index=False)
        worst_20.to_excel(writer, sheet_name="worst_20", index=False)

    print("✅ Saved:", OUT_FINAL_TRAIN_PRED_CSV)
    print("✅ Saved:", OUT_FINAL_TRAIN_EVAL_XLSX)
    print("\nFINAL TRAIN SUMMARY:\n", summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

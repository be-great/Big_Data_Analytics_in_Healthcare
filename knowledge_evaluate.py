# -----------------------
# Purpose: evaluate knowledge base (KB) retrieval performance using FAISS embeddings.
#
# Objects:
#   - SentenceTransformer: converts text queries to embeddings.
#   - FAISS indexes: vector indexes of KB facts for retrieval.
#   - Pandas DataFrames: KB facts loaded from Parquet files.
#   - Metrics: precision@k, recall@k, MRR, nDCG@k, top-k distances, Cohen's d, separability, AUC.
#   - KBs: regular KBs (patients, doctors, appointments, treatments, billing) and analytic KBs (derived statistics).
#
# Flow:
#   1) Load each KB (Parquet + FAISS index + facts).
#   2) Generate queries for in-KB evaluation.
#   3) Compute retrieval metrics (precision@k, recall@k, MRR, nDCG@k) for in-KB queries.
#   4) Generate out-of-KB queries and check abstention behavior.
#   5) Evaluate analytic KBs (without Parquet files) using distances, Cohen's d, separability, AUC.
#   6) Aggregate results and return metrics/dataframes.
# -----------------------

import pandas as pd
import numpy as np
import os
import faiss
from sentence_transformers import SentenceTransformer
import random

# -------------------------
# GLOBALS
# -------------------------
BASE_KB = "data/output/kb/"
BASE_PARQUET = "data/curated_parquet/"
TOP_K = 5  # global k
NUM_SAMPLES = 50  # sample per KB
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

KB_INDEXES = {}
KB_TEXTS = {}
KB_IDS = {}
KB_DFS = {}

ALL_KBS = ["patients", "doctors", "appointments", "treatments", "billing"]
ANALYTIC_KBS = ["gender_comparison","hospital_experience","top_specializations",
    "top_reasons_for_visits","expensive_treatment"]
# -------------------------
# LOAD KB
# -------------------------
"""
load_kb: function that loads a knowledge base
name: name of the KB (e.g., "patients", "doctors")
return: Pandas DataFrame of KB facts
"""
def load_kb(name):
    KB_INDEXES[name] = faiss.read_index(BASE_KB + f"{name}.index")
    KB_TEXTS[name] = np.load(BASE_KB + f"{name}_facts.npy", allow_pickle=True)
    df = pd.read_parquet(BASE_PARQUET + f"{name}.parquet").fillna("")
    id_col = df.columns[0]
    KB_IDS[name] = list(df[id_col].astype(str))
    KB_DFS[name] = df
    return df

# -------------------------
# METRICS
# -------------------------
"""
precision_at_k: function that computes precision at k for a single query
indices: list of retrieved indices
true_idx: true index of the fact
k: top-k cutoff
return: float (0.0 or 1.0)
"""
def precision_at_k(indices, true_idx, k):
    return 1.0 if true_idx in indices[:k] else 0.0

"""
recall_at_k: function that computes recall at k for a single query
indices: list of retrieved indices
true_idx: true index of the fact
k: top-k cutoff
return: float (0.0 or 1.0)
"""
def recall_at_k(indices, true_idx, k):
    return 1.0 if true_idx in indices[:k] else 0.0

"""
mrr: function that computes mean reciprocal rank for a single query
indices: list of retrieved indices
true_idx: true index of the fact
return: float (reciprocal rank)
"""
def mrr(indices, true_idx):
    for rank, idx in enumerate(indices, start=1):
        if idx == true_idx:
            return 1.0 / rank
    return 0.0
"""
ndcg_at_k: function that computes normalized discounted cumulative gain at k
indices: list of retrieved indices
true_idx: true index of the fact
k: top-k cutoff
return: float (nDCG score)
"""
def ndcg_at_k(indices, true_idx, k):
    if true_idx not in indices[:k]:
        return 0.0
    rank = indices[:k].index(true_idx) + 1
    return 1.0 / np.log2(rank + 1)

# -------------------------
# GENERATE QUERY
# -------------------------
"""
generate_query: function that creates a textual query for a KB row
row: Pandas Series representing a KB fact
name: KB name
return: string query
"""
def generate_query(row, name):
    if name == "patients":
        return f"Patient {row['patient_first_name']} {row['patient_last_name']} with id {row['patient_id']} lives at {row['address']}."
    if name == "doctors":
        return f"Doctor {row['doctor_first_name']} {row['doctor_last_name']} with id {row['doctor_id']} specializes in {row['specialization']}."
    if name == "appointments":
        return f"Appointment {row['appointment_id']} with doctor {row['doctor_id']} and patient {row['patient_id']} on {row['appointment_date']}."
    if name == "treatments":
        return f"Treatment {row['treatment_id']} costing {row['cost']} for appointment {row['appointment_id']}."
    if name == "billing":
        return f"Bill {row['bill_id']} for patient {row['patient_id']} amount {row['amount']}."
    return ""

# -------------------------
# KB SELECTION (for out-of-KB queries)
# -------------------------
"""
get_kb_name: function that guesses KB name from query text
query: text query
return: KB name string or None
"""
def get_kb_name(query):
    q = query.lower()
    if any(word in q for word in ["doctor", "specialization", "years experience"]):
        return "doctors"
    elif any(word in q for word in ["appointment", "visit", "schedule"]):
        return "appointments"
    elif any(word in q for word in ["treatment", "cost", "description"]):
        return "treatments"
    elif any(word in q for word in ["bill", "payment", "amount"]):
        return "billing"
    elif any(word in q for word in ["patient"]):
        return "patients"
    return None

# -------------------------
# RETRIEVE FROM KB
# -------------------------
"""
retrieve_from_kb: function that retrieves top-k facts from a KB
query: text query
kb_name: KB name
top_k: number of results
return: list of retrieved texts, list of indices, list of distances
"""
def retrieve_from_kb(query, kb_name, top_k=TOP_K):
    q_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    D, I = KB_INDEXES[kb_name].search(q_vec, top_k)
    return [KB_TEXTS[kb_name][i] for i in I[0]], I[0], D[0]

# -------------------------
# EVALUATION
# -------------------------
"""
evaluate_all_kbs: function that evaluates all regular KBs
all_kbs: list of KB names
top_k: top-k retrieval
num_samples: number of sample queries per KB
return: dataframe of results, overall in-KB metrics, abstention accuracy for out-of-KB queries
"""
def evaluate_all_kbs(all_kbs, top_k=TOP_K, num_samples=NUM_SAMPLES):
    global KB_INDEXES, KB_TEXTS, KB_IDS, KB_DFS
    all_results = []
    out_of_kb_correct = 0
    out_of_kb_total = 0

    # --- Ensure all KBs are loaded ---
    for kb_name in all_kbs:
        if kb_name not in KB_INDEXES:
            load_kb(kb_name)

    for name in all_kbs:
        print(f"Evaluating KB: {name}")
        df = KB_DFS[name]
        df_sample = df.sample(min(num_samples, len(df)))
        index = KB_INDEXES[name]
        id_list = KB_IDS[name]

        # --- In-KB queries ---
        for _, row in df_sample.iterrows():
            query = generate_query(row, name)
            true_id = str(row[df.columns[0]])
            true_idx = id_list.index(true_id)

            q_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
            D, I = index.search(q_vec, top_k)
            indices = list(I[0])
            distances = list(D[0])

            all_results.append({
                "kb": name,
                "query": query,
                "true_id": true_id,
                "top1_distance": distances[0],
                "mean_topk_distance": float(np.mean(distances)),
                "precision@k": precision_at_k(indices, true_idx, top_k),
                "recall@k": recall_at_k(indices, true_idx, top_k),
                "mrr": mrr(indices, true_idx),
                "ndcg@k": ndcg_at_k(indices, true_idx, top_k),
                "type": "in_kb"
            })

        # --- Out-of-KB queries (diverse) ---
        for _ in range(num_samples // 2):
            kb_type = random.choice(all_kbs)
            if kb_type == "patients":
                out_query = f"Patient RANDOM_{random.randint(1000,9999)} not in KB"
            elif kb_type == "doctors":
                out_query = f"Doctor RANDOM_{random.randint(1000,9999)} not in KB"
            elif kb_type == "appointments":
                out_query = f"Appointment RANDOM_{random.randint(1000,9999)} not in KB"
            elif kb_type == "treatments":
                out_query = f"Treatment RANDOM_{random.randint(1000,9999)} not in KB"
            elif kb_type == "billing":
                out_query = f"Bill RANDOM_{random.randint(1000,9999)} not in KB"
            else:
                continue

            kb = get_kb_name(out_query)
            if kb:
                retrieved_texts, retrieved_indices, distances = retrieve_from_kb(out_query, kb, top_k)
                if all([dist > 1.0 for dist in distances]):  # assume "not found"
                    out_of_kb_correct += 1
                out_of_kb_total += 1
                all_results.append({
                    "kb": kb,
                    "query": out_query,
                    "true_id": None,
                    "top1_distance": distances[0],
                    "mean_topk_distance": float(np.mean(distances)),
                    "precision@k": 0.0,
                    "recall@k": 0.0,
                    "mrr": 0.0,
                    "ndcg@k": 0.0,
                    "type": "out_of_kb"
                })

    df_all = pd.DataFrame(all_results)
    overall_metrics = df_all[df_all["type"]=="in_kb"][["precision@k","recall@k","mrr","ndcg@k","top1_distance","mean_topk_distance"]].mean()
    abstention_acc = out_of_kb_correct / out_of_kb_total if out_of_kb_total > 0 else None
    KB_INDEXES = {}
    KB_TEXTS = {}
    KB_IDS = {}
    KB_DFS = {}
    return df_all, overall_metrics, abstention_acc

"""
evaluate_analytic_kbs: function that evaluates analytic KBs (without Parquet)
ks: list of analytic KB names
num_noise: number of random noise vectors
num_rand_text: number of random text queries
return: dataframe of metrics per analytic KB
"""
def evaluate_analytic_kbs(ks, num_noise=200, num_rand_text=100):
    rows = []
    for name in ks:
        parquet_path = BASE_PARQUET + f"{name}.parquet"
        if os.path.exists(parquet_path):
            # skip regular KBs
            continue

        # ensure index/texts loaded
        if name not in KB_INDEXES:
            KB_INDEXES[name] = faiss.read_index(BASE_KB + f"{name}.index")
        if name not in KB_TEXTS:
            KB_TEXTS[name] = np.load(BASE_KB + f"{name}_facts.npy", allow_pickle=True)

        index = KB_INDEXES[name]
        facts = list(KB_TEXTS[name].astype(str)) if len(KB_TEXTS[name])>0 else [""]

        # --- in-KB distances: query each fact text ---
        in_top1 = []
        for f in facts:
            qv = embedder.encode([f], convert_to_numpy=True).astype("float32")
            D, I = index.search(qv, 1)
            in_top1.append(float(D[0][0]))

        # --- noise vectors (random normal) ---
        # infer embedding dim
        if hasattr(index, "d"):
            d = index.d
        else:
            d = embedder.encode(["x"]).shape[1]
        noise_vecs = np.random.normal(size=(num_noise, d)).astype("float32")
        Dn, In = index.search(noise_vecs, 1)
        noise_top1 = [float(x[0]) for x in Dn]

        # --- random textual queries ---
        randq = []
        for _ in range(num_rand_text):
            s = f"RANDOM_{random.randint(100000,999999)}"
            qv = embedder.encode([s], convert_to_numpy=True).astype("float32")
            Dq, Iq = index.search(qv, 1)
            randq.append(float(Dq[0][0]))

        # --- stats helper ---
        def stats(a):
            a = np.array(a)
            return float(a.mean()), float(np.median(a)), float(a.std(ddof=1) if len(a)>1 else 0.0), len(a)

        in_mean, in_med, in_std, in_n = stats(in_top1)
        noise_mean, noise_med, noise_std, noise_n = stats(noise_top1)
        rand_mean, rand_med, rand_std, rand_n = stats(randq)

        # Cohen's d (noise vs in): (noise_mean - in_mean) / pooled_std
        pooled_var = (((in_n-1)*(in_std**2 if in_n>1 else 0) + (noise_n-1)*(noise_std**2 if noise_n>1 else 0))
                      / max(in_n+noise_n-2, 1))
        pooled_std = np.sqrt(pooled_var) if pooled_var>0 else 1e-9
        cohen_d = (noise_mean - in_mean) / pooled_std

        # separability: fraction of noise distances > (in_mean + 2*in_std)
        threshold = in_mean + 2 * in_std
        separable_pct = float(np.mean([1.0 if v > threshold else 0.0 for v in noise_top1])) if noise_n>0 else None

        # AUC (pairwise comparator): probability in < noise
        def auc_from_scores(pos, neg):
            if len(pos)==0 or len(neg)==0:
                return None
            U=0.0
            for p in pos:
                for n in neg:
                    if p < n:
                        U += 1.0
                    elif p==n:
                        U += 0.5
            return U / (len(pos)*len(neg))
        auc_noise = auc_from_scores(in_top1, noise_top1)
        auc_randq = auc_from_scores(in_top1, randq)

        rows.append({
            "kb": name,
            "n_facts": in_n,
            "in_mean": in_mean,
            "in_median": in_med,
            "in_std": in_std,
            "noise_mean": noise_mean,
            "noise_median": noise_med,
            "noise_std": noise_std,
            "randq_mean": rand_mean,
            "cohen_d_noise": cohen_d,
            "separable_pct_noise": separable_pct,
            "auc_noise": auc_noise,
            "auc_rand_text": auc_randq,
            "in_top1_list": in_top1,
            "noise_top1_list_sample": noise_top1[:50],   # keep output small
            "rand_text_top1_list": randq
        })

    return pd.DataFrame(rows)
# -------------------------
# RUN
# -------------------------
df_results, overall_metrics, abstention_accuracy = evaluate_all_kbs(ALL_KBS, top_k=TOP_K, num_samples=NUM_SAMPLES)

print("Overall In-KB Metrics:\n", overall_metrics)
print("Abstention Accuracy (Out-of-KB Queries):", abstention_accuracy)
df_analytic = evaluate_analytic_kbs(ANALYTIC_KBS, num_noise=500, num_rand_text=200)
print("analytic_kbs:")
print(df_analytic[["kb","n_facts","in_mean","noise_mean","cohen_d_noise","separable_pct_noise","auc_noise"]])

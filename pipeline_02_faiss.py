# -----------------------
# Purpose: build FAISS knowledge bases from hospital datasets and analysis results.
#
# Objects:
#   - SparkSession: used to load and process DataFrames.
#   - Input Parquet tables: patients, doctors, appointments, treatments, billing.
#   - SentenceTransformer model: converts text into embeddings.
#   - build_kb(df, name, text_func): creates FAISS index from DataFrame and saves texts/index.
#   - FAISS index: stores embeddings for fast similarity search.
#   - Analysis functions: gender_distribution, most_experienced_branch, top_specialization, common_visit_reason, treatments_by_cost.
#
# Flow:
#   1) Start Spark session.
#   2) Load parquet datasets and analysis results.
#   3) Initialize SentenceTransformer model.
#   4) Build FAISS knowledge bases for each dataset.
#   5) Build FAISS knowledge bases for analysis insights.
#   6) Save FAISS indices and corresponding texts to output folder.
#   7) Stop Spark session.
# -----------------------

import argparse, os
import numpy as np
import torch, faiss
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer
from data_analysis import load_data, gender_distribution, most_experienced_branch, top_specialization, common_visit_reason, treatments_by_cost

base = "data/curated_parquet"
out = "data/output/kb"
os.makedirs(out, exist_ok=True)

spark = SparkSession.builder.appName("BuildFAISS").getOrCreate()

# Load parquet files
patients, doctors, appts, treats, billing = load_data()
# load analysis outputs
gender = gender_distribution(patients)
hos_branch = most_experienced_branch(doctors)
top_spe = top_specialization(doctors)
top_com_reason = common_visit_reason(appts)
exp_tre = treatments_by_cost(billing, treats)
""""To convert text sentences into numerical embeddings"""
# Load SentenceTransformer model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
"""
build_kb: function that creates a FAISS knowledge base from a Spark DataFrame
df: input Spark DataFrame
name: output knowledge base name
text_func: function converting DataFrame row into text string
return: None
Short: encodes rows, builds FAISS index, saves index and texts
"""
def build_kb(df, name, text_func):
    rows = df.collect()
    texts, ids = [], []
    for r in rows:
        txt = text_func(r)
        texts.append(txt)
        ids.append(str(r[0]))  # first column = id
    # The encoding step
    emb = model.encode(texts, batch_size=256, convert_to_numpy=True, show_progress_bar=True)
    d = emb.shape[1]
    # Build FAISS index
    res = faiss.StandardGpuResources() if torch.cuda.is_available() else None
    index = faiss.IndexFlatL2(d)
    if res:
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(emb.astype(np.float32))
    # Save FAISS index and texts
    faiss.write_index(index, f"{out}/{name}.index")
    np.save(f"{out}/{name}_facts.npy", np.array(texts))
    print(f"✅ {name.capitalize()} knowledge base saved ({len(texts)} facts)")

# ---- Define text builders for each dataset ----
build_kb(
    patients, "patients",
    lambda r: f"Patient {r['patient_first_name']} {r['patient_last_name']} with the id {r['patient_id']} ({r['gender']}) born {r['date_of_birth']} lives at {r['address']} with insurance from {r['insurance_provider']}."
)

build_kb(
    doctors, "doctors",
    lambda r: f"Doctor {r['doctor_first_name']} {r['doctor_last_name']} with the id {r['doctor_id']} specializes in {r['specialization']} with {r['years_experience']} years experience at {r['hospital_branch']}."
)

build_kb(
    appts, "appointments",
    lambda r: f"Appointment {r['appointment_id']} on {r['appointment_date']} at {r['appointment_time']} for reason {r['reason_for_visit']} with doctor {r['doctor_id']} and patient {r['patient_id']}."
)

build_kb(
    treats, "treatments",
    lambda r: f"Treatment {r['treatment_id']} of type {r['treatment_type']} for appointment id {r['appointment_id']} described as {r['description']} costing {r['cost']} on {r['treatment_date']}."
)

build_kb(
    billing, "billing",
    lambda r: f"Billing record {r['bill_id']} with the patient id {r['patient_id']} dated {r['bill_date']} amount {r['amount']} paid via {r['payment_method']} (status {r['payment_status']})."
)


"""
Kb for analysis contents
"""

# ------------------ GENDER DISTRIBUTION ------------------
gender = gender_distribution(patients)

gender_rows = gender.collect()
total_visits = sum(r["count"] for r in gender_rows)

gender = {r["gender"]: round((r["count"] / total_visits) * 100, 2) for r in gender_rows}

genders = list(gender.keys())
g1, g2 = genders[0], genders[1]

gender_diff_text = (
    f"{g1} visits represent {gender[g1]}% compared to {gender[g2]}% for {g2}."
)

build_kb(
    spark.createDataFrame([(0,)], ["id"]),
    "gender_comparison",
    lambda r: gender_diff_text
)


# ------------------ MOST EXPERIENCED BRANCH ------------------
branch_rows = hos_branch.collect()
branch_names = [r["hospital_branch"] for r in branch_rows]

branch_text = (
    "Hospital branches ranked by most experienced doctors: "
    + ", ".join(branch_names)
    + "."
)

build_kb(
    spark.createDataFrame([(0,)], ["id"]),
    "hospital_experience",
    lambda r: branch_text
)


# ------------------ TOP SPECIALIZATIONS ------------------
spe_rows = top_spe.collect()
top_spe_list = [r["specialization"] for r in spe_rows[:5]]

specializations_text = (
    "Top dominating specializations: " + ", ".join(top_spe_list) + "."
)

build_kb(
    spark.createDataFrame([(0,)], ["id"]),
    "top_specializations",
    lambda r: specializations_text
)


# ------------------ MOST COMMON REASONS FOR VISITS ------------------
reason_rows = top_com_reason.collect()
top_reason_list = [r["reason_for_visit"] for r in reason_rows[:5]]

reasons_text = (
    "Most common reasons for visits: " + ", ".join(top_reason_list) + "."
)

build_kb(
    spark.createDataFrame([(0,)], ["id"]),
    "top_reasons_for_visits",
    lambda r: reasons_text
)


# ------------------ MOST EXPENSIVE TREATMENT ------------------
exp_rows = exp_tre.collect()
exp_name = exp_rows[0]["treatment_type"]
exp_cost = round(exp_rows[0]["avg_cost"], 2)

build_kb(
    spark.createDataFrame([(0,)], ["id"]),
    "expensive_treatment",
    lambda r: f"The treatment '{exp_name}' is the most expensive, with an average cost of ${exp_cost}."
)


spark.stop()

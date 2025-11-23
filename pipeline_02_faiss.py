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
"""Function to build a knowledge base"""
# turns each DataFrame row into a text
# sentence and creates a corresponding
# unique ID list. Example : patient Ali Bob
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

""""
Kb for analysis contents
"""

# Gender comparison KB
gender_diff_text = f"{list(gender.keys())[0]} visits are {gender[list(gender.keys())[0]][0]}% higher than {gender[list(gender.keys())[0]][1]} visits."
build_kb(
    spark.createDataFrame([(0,)], ["id"]), "gender_comparison",  # dummy DF with one row
    lambda r: gender_diff_text
)

# Hospital branches with most experienced doctors
branch_text = f"Hospital branches that have the most experienced doctors are: {', '.join(hos_branch.keys())}."
build_kb(
    spark.createDataFrame([(0,)], ["id"]),
    "hospital_experience",
    lambda r: branch_text
)

# Top dominating specializations
specializations_text = f"Top dominating specializations: {', '. join(top_spe[:5])}."
build_kb(
    spark.createDataFrame([(0,)], ["id"]),
    "top_specializations",
    lambda r: specializations_text
)

# Most common reasons for visits
reasons_text = f"Most common reasons for visits: {', '.join(top_com_reason[:5])}."
build_kb(
    spark.createDataFrame([(0,)], ["id"]),
    "top_reasons_for_visits",
    lambda r: reasons_text
)

# Most expensive treatment KB
exp_treatment_name = list(exp_tre.keys())[0]
exp_treatment_cost = list(exp_tre.values())[0]
build_kb(
    spark.createDataFrame([(0,)], ["id"]),
    "expensive_treatment",
    lambda r: f"The treatment '{exp_treatment_name}' is the most expensive, with an average cost of ${exp_treatment_cost}."
)
spark.stop()

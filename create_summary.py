"""
Answer those scenarios
A — Which gender goes to the hospital more?
B — Which hospital branch has the most experienced doctors?
C — Which specialization dominates the others?
D — What is the most common reason for visits?
E — What is the ranking of treatments by cost?

"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg

# Start Spark
spark = SparkSession.builder.appName("HospitalAnalysis").getOrCreate()

# ===== 1️⃣ Read the Parquet files =====
base = f"data/curated_parquet"

patients_c  = spark.read.parquet(f"{base}/patients.parquet")
doctors_c   = spark.read.parquet(f"{base}/doctors.parquet")
appts_c     = spark.read.parquet(f"{base}/appointments.parquet")
treats_c    = spark.read.parquet(f"{base}/treatments.parquet")
bill_c      = spark.read.parquet(f"{base}/billing.parquet")

# ===== 2️⃣ A: Which gender goes to the hospital more =====
print("\n🅰️ Which gender goes to the hospital more:")
patients_c.groupBy("gender").count().orderBy("count", ascending=False).show()

# ===== 3️⃣ B: Which hospital branch has the most experienced doctors =====
print("\n🅱️ Which hospital branch has the most experienced doctors:")
doctors_c.groupBy("hospital_branch") \
         .agg(avg("years_experience").alias("avg_experience")) \
         .orderBy(col("avg_experience").desc()).show()

# ===== 4️⃣ C: Which specialization dominates others =====
print("\n🅲 Which specialization dominates others:")
doctors_c.groupBy("specialization").count().orderBy("count", ascending=False).show()

# ===== 5️⃣ D: What is the most common reason for visits =====
print("\n🅳 What is the most common reason for visits:")
appts_c.groupBy("reason_for_visit").count().orderBy("count", ascending=False).show()

# ===== 6️⃣ E: Rank treatments by average cost =====
print("\n🅴 Rank treatments by average cost:")
bill_c.join(treats_c, "treatment_id", "inner") \
      .groupBy("treatment_type") \
      .agg(avg("cost").alias("avg_cost")) \
      .orderBy(col("avg_cost").desc()).show()

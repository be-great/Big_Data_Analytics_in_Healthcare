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


def load_data():
      # Start Spark
      spark = SparkSession.builder.appName("HospitalAnalysis").getOrCreate()

      # ===== 1️ Read the Parquet files =====
      base = f"data/curated_parquet"

      patients_c  = spark.read.parquet(f"{base}/patients.parquet").na.fill("")
      doctors_c   = spark.read.parquet(f"{base}/doctors.parquet").na.fill("")
      appts_c     = spark.read.parquet(f"{base}/appointments.parquet").na.fill("")
      treats_c    = spark.read.parquet(f"{base}/treatments.parquet").na.fill("")
      bill_c      = spark.read.parquet(f"{base}/billing.parquet").na.fill("")
      # return the parquet files that needed 
      return patients_c, doctors_c, appts_c, treats_c, bill_c

def gender_distribution(patients_c):
    """Scenario A: Which gender goes to the hospital more."""
    return patients_c.groupBy("gender").count().orderBy("count", ascending=False)

def most_experienced_branch(doctors_c):
    """Scenario B: Which hospital branch has the most experienced doctors."""
    return doctors_c.groupBy("hospital_branch") \
                    .agg(avg("years_experience").alias("avg_experience")) \
                    .orderBy(col("avg_experience").desc())

def top_specialization(doctors_c):
    """Scenario C: Which specialization dominates others."""
    return doctors_c.groupBy("specialization").count().orderBy("count", ascending=False)

def common_visit_reason(appts_c):
    """Scenario D: Most common reason for visits."""
    return appts_c.groupBy("reason_for_visit").count().orderBy("count", ascending=False)

def treatments_by_cost(bill_c, treats_c):
    """Scenario E: Rank treatments by average cost."""
    return bill_c.join(treats_c, "treatment_id", "inner") \
                 .groupBy("treatment_type") \
                 .agg(avg("cost").alias("avg_cost")) \
                 .orderBy(col("avg_cost").desc())

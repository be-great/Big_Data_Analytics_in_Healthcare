# -----------------------
# Purpose: Analyze hospital datasets to answer common scenarios and generate insights.
#
# Objects:
#   - SparkSession: used to read Parquet files and perform DataFrame operations.
#   - Parquet files: patients, doctors, appointments, treatments, billing.
#   - Functions:
#       • load_data(): loads all hospital Parquet datasets.
#       • gender_distribution(patients_c): counts patients per gender.
#       • most_experienced_branch(doctors_c): computes average doctor experience per hospital branch.
#       • top_specialization(doctors_c): counts number of doctors per specialization.
#       • common_visit_reason(appts_c): counts most common reasons for visits.
#       • treatments_by_cost(bill_c, treats_c): ranks treatments by average cost.
#
# Flow:
#   1) Start Spark session.
#   2) Load Parquet datasets via load_data().
#   3) Compute analysis per scenario:
#       - Scenario A: gender_distribution()
#       - Scenario B: most_experienced_branch()
#       - Scenario C: top_specialization()
#       - Scenario D: common_visit_reason()
#       - Scenario E: treatments_by_cost()
#   4) Return Spark DataFrames with summarized results for each scenario.
# -----------------------

"""
The script answer those scenarios
A — Which gender goes to the hospital more?
B — Which hospital branch has the most experienced doctors?
C — Which specialization dominates the others?
D — What is the most common reason for visits?
E — What is the ranking of treatments by cost?

"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg


"""
load_data: function that loads hospital parquet datasets
return: patients_c, doctors_c, appts_c, treats_c, bill_c (Spark DataFrames)
"""
def load_data():
      # Start Spark
      spark = SparkSession.builder.appName("HospitalAnalysis").getOrCreate()

      # ===== 1️ Read the Parquet files =====
      base = f"data/curated_parquet"
      #base = "../hospital_record/"
      patients_c  = spark.read.parquet(f"{base}/patients.parquet").na.fill("")
      doctors_c   = spark.read.parquet(f"{base}/doctors.parquet").na.fill("")
      appts_c     = spark.read.parquet(f"{base}/appointments.parquet").na.fill("")
      treats_c    = spark.read.parquet(f"{base}/treatments.parquet").na.fill("")
      bill_c      = spark.read.parquet(f"{base}/billing.parquet").na.fill("")
      # return the parquet files that needed 
      return patients_c, doctors_c, appts_c, treats_c, bill_c
"""
gender_distribution: function that computes gender counts
patients_c: patients DataFrame
return: Spark DataFrame with columns [gender, count]
"""
def gender_distribution(patients_c):
    """Scenario A: Which gender goes to the hospital more."""
    return patients_c.groupBy("gender").count().orderBy("count", ascending=False)

"""
most_experienced_branch: function that computes average doctor experience per branch
doctors_c: doctors DataFrame
return: Spark DataFrame with columns [hospital_branch, avg_experience]
"""
def most_experienced_branch(doctors_c):
    """Scenario B: Which hospital branch has the most experienced doctors."""
    return doctors_c.groupBy("hospital_branch") \
                    .agg(avg("years_experience").alias("avg_experience")) \
                    .orderBy(col("avg_experience").desc())
"""
top_specialization: function that counts doctors per specialization
doctors_c: doctors DataFrame
return: Spark DataFrame with columns [specialization, count]
"""
def top_specialization(doctors_c):
    """Scenario C: Which specialization dominates others."""
    return doctors_c.groupBy("specialization").count().orderBy("count", ascending=False)

"""
common_visit_reason: function that counts appointment reasons
appts_c: appointments DataFrame
return: Spark DataFrame with columns [reason_for_visit, count]
"""
def common_visit_reason(appts_c):
    """Scenario D: Most common reason for visits."""
    return appts_c.groupBy("reason_for_visit").count().orderBy("count", ascending=False)


"""
treatments_by_cost: function that ranks treatments by average cost
bill_c: billing DataFrame
treats_c: treatments DataFrame
return: Spark DataFrame with columns [treatment_type, avg_cost]
"""
def treatments_by_cost(bill_c, treats_c):
    """Scenario E: Rank treatments by average cost."""
    return bill_c.join(treats_c, "treatment_id", "inner") \
                 .groupBy("treatment_type") \
                 .agg(avg("cost").alias("avg_cost")) \
                 .orderBy(col("avg_cost").desc())

# patients_c, doctors_c, appts_c, treats_c, bill_c= load_data()
# gender_distribution(patients_c)
# most_experienced_branch(doctors_c)
# top_specialization(doctors_c)
# common_visit_reason(appts_c)
# treatments_by_cost(bill_c, treats_c)
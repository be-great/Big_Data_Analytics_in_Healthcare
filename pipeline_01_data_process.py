# -----------------------
# Purpose: load CSV hospital tables, clean them, and save each as a single Parquet file.
#
# Objects:
#   - SparkSession: used to read CSV files and process DataFrames.
#   - Input CSV files: patients, doctors, appointments, treatments, billing.
#   - clean(df): cleans string columns (lower + trim) and removes duplicates.
#   - save_single_parquet(df, out_dir, name): writes a single Parquet file safely using DuckDB.
#   - DuckDB engine: copies Parquet to final file to avoid Spark timestamp issues.
#
# Flow:
#   1) Parse input/output arguments.
#   2) Start Spark session.
#   3) Read CSV tables.
#   4) Clean each table and fix duplicate column names.
#   5) Save each table as one Parquet file.
#   6) Stop Spark.
# -----------------------

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, trim
import os, shutil
import duckdb
import os
import shutil
import duckdb


parser = argparse.ArgumentParser()
parser.add_argument("--inbase", required=True, help="Base dir containing data_csv (local or hdfs://)")
parser.add_argument("--out", required=True, help="Output Parquet folder (local or hdfs:///)")
args = parser.parse_args()

spark = SparkSession.builder.appName("HospitalCurate").getOrCreate()


"""
save_single_parquet: function that writes a DataFrame as one Parquet file
df: input Spark DataFrame
out_dir: folder where the parquet should be saved
name: final parquet file name (without extension)
return: None
"""
def save_single_parquet(df, out_dir, name):
    temp_path = os.path.join(out_dir, "tmp_folder")
    os.makedirs(temp_path, exist_ok=True)

    # Write Spark DataFrame to temporary folder (coalesce to 1 partition)
    df.coalesce(1).write.mode("overwrite").parquet(temp_path)

    # Find the single part file
    part_file = [f for f in os.listdir(temp_path) if f.endswith(".parquet")][0]
    input_file = os.path.join(temp_path, part_file)

    # Final file path
    final_file = os.path.join(out_dir, f"{name}.parquet")

    # Use DuckDB to copy Parquet (handles timestamps/INT64 safely)
    duckdb.sql(f"COPY (SELECT * FROM '{input_file}') TO '{final_file}' (FORMAT PARQUET)")

    # Remove temporary folder
    shutil.rmtree(temp_path)

    print(f"Saved single Parquet file at: {final_file}")
"""
read_csv: function that reads one CSV file into a Spark DataFrame
name: CSV file name without .csv
return: Spark DataFrame
"""
def read_csv(name):
    return (spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(f"{args.inbase}/{name}.csv"))

patients     = read_csv("patients")
doctors      = read_csv("doctors")
appointments = read_csv("appointments")
treatments   = read_csv("treatments")
billing      = read_csv("billing")

"""
clean: function that trims and lowercases all string columns
df: input Spark DataFrame
return: cleaned Spark DataFrame
"""
def clean(df):
    # lower+trim only for string columns
    return df.select([lower(trim(c)).alias(c) if t.simpleString()=="string" else col(c)
                      for c,t in zip(df.columns,[f.dataType for f in df.schema.fields])])

# lower case , remove unwanted symobles and remove duplicates across all columns
patients_c = clean(patients).dropDuplicates()
doctors_c  = clean(doctors).dropDuplicates()
appts_c    = clean(appointments).dropDuplicates()
treats_c   = clean(treatments).dropDuplicates()
bill_c     = clean(billing).dropDuplicates()


# Rename duplicate columns before join
patients_c = patients_c.withColumnRenamed("email", "patient_email")
doctors_c  = doctors_c.withColumnRenamed("email", "doctor_email")
patients_c = patients_c.withColumnRenamed("first_name", "patient_first_name")
patients_c = patients_c.withColumnRenamed("last_name", "patient_last_name")
doctors_c  = doctors_c.withColumnRenamed("first_name", "doctor_first_name")
doctors_c  = doctors_c.withColumnRenamed("last_name", "doctor_last_name")

# create the parquet
base = f"{args.out}/"
save_single_parquet(patients_c, base, "patients")
save_single_parquet(doctors_c, base, "doctors")
save_single_parquet(appts_c, base, "appointments")
save_single_parquet(treats_c, base, "treatments")
save_single_parquet(bill_c, base, "billing")

spark.stop()


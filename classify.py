from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    year, current_date, count, avg, col, max, add_months, when, rand
)
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from functools import reduce
from pyspark.sql.functions import month, year, count
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

# Start Spark
spark = SparkSession.builder.appName("HospitalML").getOrCreate()
base = "data/curated_parquet"

# Load datasets
patients_c = spark.read.parquet(f"{base}/patients.parquet")
appts_c    = spark.read.parquet(f"{base}/appointments.parquet")
treats_c   = spark.read.parquet(f"{base}/treatments.parquet")
bill_c     = spark.read.parquet(f"{base}/billing.parquet")


"""Use discriptive analysis"""
visits_per_month = appts_c.groupBy(year("appointment_date").alias("year"),
                                   month("appointment_date").alias("month")) \
                          .agg(count("patient_id").alias("num_patients"))
visits_per_month.show()
"""Analysis the model using linear regression"""
assembler = VectorAssembler(inputCols=["month", "year"], outputCol="features")
data = assembler.transform(visits_per_month).select("features", "num_patients")

lr = LinearRegression(labelCol="num_patients", featuresCol="features")
model = lr.fit(data)
predictions = model.transform(data)
predictions.show()
"""Evaluate the model"""
# Evaluate the model
# Split data: 80% train, 20% test
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Fit the model on training data
lr = LinearRegression(labelCol="num_patients", featuresCol="features")
model = lr.fit(train_data)

# Make predictions on test data
predictions = model.transform(test_data)

# Evaluate on test data
from pyspark.ml.evaluation import RegressionEvaluator

evaluator_rmse = RegressionEvaluator(labelCol="num_patients", predictionCol="prediction", metricName="rmse")
evaluator_r2   = RegressionEvaluator(labelCol="num_patients", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)
print(f"📊 Evaluation Metrics:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
""" 0.34 = %34 """
print(f"R² (Coefficient of Determination): {r2:.2f}")
from pyspark.sql import SparkSession
from pyspark.sql.functions import trunc, countDistinct, col, month as month_func, lag
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql.types import StructType, StructField
import numpy as np

# ------------------ Start Spark ------------------
spark = SparkSession.builder.appName("HospitalNext6MonthsPrediction_Fixed").getOrCreate()
base = "data/curated_parquet"

# ------------------ Load dataset ------------------
appts_c = spark.read.parquet(f"{base}/appointments.parquet")

# ------------------ Aggregate monthly patient counts ------------------
monthly_df = appts_c.groupBy(trunc("appointment_date", "month").alias("month")) \
                    .agg(countDistinct("patient_id").alias("num_patients")) \
                    .orderBy("month")

# ------------------ Add lag features ------------------
"""lag: something happend before : in the previous 2 months"""
window_spec = Window.orderBy("month")
for i in range(1, 3):
    monthly_df = monthly_df.withColumn(f"prev_{i}", lag("num_patients", i).over(window_spec))

# Drop first 2 rows with nulls
monthly_df = monthly_df.na.drop(subset=["prev_1", "prev_2"])

# ------------------ Add seasonal feature ------------------
"""convert month to number : 1-12"""
monthly_df = monthly_df.withColumn("month_num", month_func("month"))

# ------------------ Assemble features ------------------
feature_cols = ["prev_1", "prev_2", "month_num"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Keep lag columns for teacher forcing
data = assembler.transform(monthly_df).select(*feature_cols, "features", "num_patients", "month")

# ------------------ Safe train/test split ------------------
total_months = data.count()
n_test_months = min(6, total_months - 1) 
train_data = data.limit(total_months - n_test_months)
test_data = data.orderBy(col("month").desc()).limit(n_test_months).orderBy("month")

# ------------------ Train Random Forest ------------------
rf = RandomForestRegressor(labelCol="num_patients", featuresCol="features",
                           numTrees=100, maxDepth=5, seed=42)
model = rf.fit(train_data)
model.save("data/output/visitor_predict_model")
# ------------------ Teacher forcing test ------------------
test_rows = test_data.collect()
test_features = []
for row in test_rows:
    features_vec = Vectors.dense([row[f] for f in feature_cols])
    test_features.append((features_vec,))

# Create DataFrame safely (handles empty test)
schema = StructType([StructField("features", VectorUDT(), True)])
test_df = spark.createDataFrame(test_features, schema)

predictions_df = model.transform(test_df)

# ------------------ Collect predictions and actuals ------------------
pred_array = [row["prediction"] for row in predictions_df.collect()]
actual_array = [row["num_patients"] for row in test_rows]

# ------------------ Evaluate ------------------
test_array = np.array(actual_array)
pred_array = np.array(pred_array)

r2 = 1 - np.sum((test_array - pred_array)**2) / np.sum((test_array - np.mean(test_array))**2)
rmse = np.sqrt(np.mean((test_array - pred_array)**2))

print(":) Next 6 months prediction evaluation (fixed & safe):")
print(f"Predictions: {pred_array}")
print(f"Actual:      {test_array}")
print(f"R²:          {r2:.2f}")
print(f"RMSE:        {rmse:.2f}")

spark.stop()
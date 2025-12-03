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
# Mean Absolute Error (MAE)
mae = np.mean(np.abs(test_array - pred_array))

# Mean Absolute Percentage Error (MAPE)
# Handle zero values in actuals to avoid division by zero
mask = test_array != 0
mape = np.mean(np.abs((test_array[mask] - pred_array[mask]) / test_array[mask])) * 100 if np.any(mask) else None

# Relative Error (RMSE as % of mean)
mean_actual = np.mean(test_array)
rmse_relative = (rmse / mean_actual) * 100 if mean_actual != 0 else None

# Standard Deviation of actuals
std_actual = np.std(test_array)
print(":) Next 6 months prediction evaluation:")
print(f"Predictions: {pred_array}")
print(f"Actual:      {test_array}")
# print(f"R²:          {r2:.2f}")
# print(f"RMSE:        {rmse:.2f}")
print("-" * 50)
print(f"R²:          {r2:.4f} ({r2*100:.1f}% variance explained)")
print(f"RMSE:        {rmse:.2f} patients")
print(f"MAE:         {mae:.2f} patients")
if mape is not None:
    print(f"MAPE:        {mape:.2f}%")
if rmse_relative is not None:
    print(f"Rel. Error:  {rmse_relative:.2f}% (RMSE/Mean)")
print("-" * 50)
print(f"Statistics of Actual Values:")
print(f"  Mean:      {mean_actual:.2f} patients")
print(f"  Std Dev:   {std_actual:.2f} patients")
print(f"  Range:     [{np.min(test_array):.0f}, {np.max(test_array):.0f}] patients")
print("-" * 50)

# ------------------ Quality Assessment ------------------
if r2 >= 0.9:
    r2_quality = "EXCELLENT"
elif r2 >= 0.7:
    r2_quality = "GOOD"
elif r2 >= 0.5:
    r2_quality = "FAIR"
else:
    r2_quality = "POOR"

if rmse_relative is not None:
    if rmse_relative < 10:
        rmse_quality = "EXCELLENT"
    elif rmse_relative < 20:
        rmse_quality = "GOOD"
    elif rmse_relative < 30:
        rmse_quality = "FAIR"
    else:
        rmse_quality = "POOR"
else:
    rmse_quality = "Cannot assess (mean is zero)"

print(f"Model Quality Assessment:")
print(f"  R²: {r2_quality}")
if rmse_quality != "Cannot assess (mean is zero)":
    print(f"  Relative Error: {rmse_quality}")
else:
    print(f"  Absolute Error: RMSE = {rmse:.0f} patients")
spark.stop()

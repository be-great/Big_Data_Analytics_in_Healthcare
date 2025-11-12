from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, trunc, lag, col, sum as spark_sum
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import countDistinct

# ------------------ Start Spark ------------------
spark = SparkSession.builder.appName("HospitalNextMonthPrediction").getOrCreate()
base = "data/curated_parquet"

# ------------------ Load datasets ------------------
appts_c = spark.read.parquet(f"{base}/appointments.parquet")

# ------------------ Aggregate monthly patient counts ------------------

monthly_df = appts_c.groupBy(trunc("appointment_date", "month").alias("month")) \
                    .agg(countDistinct("patient_id").alias("num_patients")) \
                    .orderBy("month")

# ------------------ Create lag features ------------------
window_spec = Window.orderBy("month")
monthly_df = monthly_df.withColumn("prev_month_patients", lag("num_patients").over(window_spec))

# Drop first row (lag = null)
monthly_df = monthly_df.na.drop(subset=["prev_month_patients"])

# ------------------ Assemble features ------------------
assembler = VectorAssembler(inputCols=["prev_month_patients"], outputCol="features")
data = assembler.transform(monthly_df).select("features", "num_patients", "month")

# ------------------ Train model on all data except last month ------------------
train_data = data.limit(data.count() - 1)
last_month = data.orderBy(col("month").desc()).limit(1)  # for prediction

rf = RandomForestRegressor(labelCol="num_patients", featuresCol="features",
                           numTrees=100, maxDepth=5, seed=42)
model = rf.fit(train_data)

# ------------------ Predict next month ------------------
prediction = model.transform(last_month)
prediction.select("month", "prediction").show()

# ------------------ Evaluate (optional, on all data with lag) ------------------
predictions_all = model.transform(data)
evaluator_r2 = RegressionEvaluator(labelCol="num_patients",
                                   predictionCol="prediction",
                                   metricName="r2")
evaluator_rmse = RegressionEvaluator(labelCol="num_patients",
                                     predictionCol="prediction",
                                     metricName="rmse")

r2 = evaluator_r2.evaluate(predictions_all)
rmse = evaluator_rmse.evaluate(predictions_all)

print(f"📊 Evaluation Metrics:")
print(f"R²: {r2:.2f}")
print(f"RMSE: {rmse:.2f}")

spark.stop()
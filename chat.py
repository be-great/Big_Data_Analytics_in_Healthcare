# -----------------------
# Design Summary
# Purpose: Interactive domain-aware QA system that uses Spark, ML models,
#          and a LoRA-adapted LLM to answer hospital-related queries.

# Components:
#   • SparkSession for loading Parquet data and computing analytics.
#   • RandomForestRegressionModel for visitor-count prediction.
#   • LogisticRegressionModel for domain-question detection.
#   • LoRA adapted LLM (model), FAISS knowledge indexes, knowledge facts,
#     embedding model, and tokenizer loaded via load_model().
#   • Knowledge-base selector for routing queries to the correct dataset.
#   • Context retriever for fetching top-k relevant facts.
#   • Domain classifier to decide whether to attach context or not.
#   • Chat loop for receiving queries, building prompts, generating answers.

# Functions:
#   • get_kb_name(query)
#   • retrieve_context(query, top_k)
#   • is_domain_question(query, threshold)
#
# Flow:
#   1) Start Spark session.
#   2) Load ML models and LLM + embeddings + FAISS indexes.
#   3) Determine the correct knowledge base for each query.
#   4) If domain-specific:
#        - Retrieve relevant facts or perform visitor prediction.
#        - Build prompt including context.
#      Else:
#        - Build prompt without context.
#   5) Use tokenizer + LLM to generate an answer.
#   6) Clean output to remove prompt echoes and extra generations.
#   7) Repeat until user exits.
# -----------------------
from pyspark.sql import SparkSession, Row
from pyspark.ml.classification import LogisticRegressionModel
from pipeline_04_qlora import load_model
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import trunc, countDistinct

base_file = ""
# =========================
# Start Spark session
# =========================
spark = SparkSession.builder \
    .appName("DomainQuestionChecker") \
    .getOrCreate()

# =========================
# Load pre-trained models
# =========================
# Visitor prediction model
visitor_model = RandomForestRegressionModel.load(
    base_file + "data/output/visitor_predict_model"
)

# Domain classifier (logistic regression)
model_re = LogisticRegressionModel.load(base_file + "data/output/lr_model")

# LoRA-adapted LLM + embeddings + tokenizer
# load_model() must already be defined from your previous code
model, kb_indexes, kb_facts, embedder, tokenizer = load_model()

# =========================
# Knowledge base selection
# =========================
""""
get_kb_name: function that selects the knowledge-base based on keywords in the query
query : the input text to analyze
return : the selected KB name or None
"""
def get_kb_name(query):
    query = query.lower()
    
    # Domain KB selection based on keywords
    if any(word in query for word in ["doctor", "specialization", "years experience"]):
        return "doctors"
    elif any(word in query for word in ["appointment", "visit", "schedule"]):
        return "appointments"
    elif any(word in query for word in ["treatment", "cost", "description"]):
        return "treatments"
    elif any(word in query for word in ["bill", "payment", "amount"]):
        return "billing"
    elif any(word in query for word in ["patient"]):
        return "patients"
    
    # Analytic KBs
    elif "gender" in query:
        return "gender_comparison"
    elif any(word in query for word in ["experienced", "branches", "hospital branch"]):
        return "hospital_experience"
    elif "specializations" in query or "fields" in query:
        return "top_specializations"
    elif "reasons" in query or "visits" in query:
        return "top_reasons_for_visits"
    elif "expensive treatment" in query or "most expensive" in query:
        return "expensive_treatment"
    elif "visitor prediction" in query or "predict next month" in query:
        return "visitor_prediction"
    
    # No KB matched
    else:
        return None

# =========================
# Retrieve context from KB
# =========================
"""
retrieve_context: function that retrieves top-k relevant facts or runs visitor prediction
query : the user question
top_k : number of facts to return
return : retrieved facts as text (or prediction result)
"""
def retrieve_context(query, top_k):
    """
    Retrieves top-k relevant facts for a query from the appropriate KB.
    """
    q_vec = embedder.encode([query], convert_to_numpy=True)
    kb_name = get_kb_name(query)
    
    if not kb_name:
        return "I don't know :(."

    # Special case: visitor prediction using RandomForest
    if kb_name == "visitor_prediction":
        base = "data/curated_parquet"
        monthly_df = spark.read.parquet(f"{base}/appointments.parquet") \
            .groupBy(trunc("appointment_date", "month").alias("month")) \
            .agg(countDistinct("patient_id").alias("num_patients")) \
            .orderBy("month")
        
        # Get last 2 months' patient counts
        last_2 = monthly_df.orderBy("month", ascending=False).limit(2).collect()
        prev_1 = last_2[0]["num_patients"]
        prev_2 = last_2[1]["num_patients"]
        
        # Next month number (1–12)
        next_month_num = (datetime.today().month % 12) + 1
        
        # Assemble features for prediction
        features = Vectors.dense([prev_1, prev_2, next_month_num])
        pred_df = spark.createDataFrame([Row(features=features)])
        prediction = visitor_model.transform(pred_df).collect()[0]["prediction"]
        return f"Expected visitors next month: {int(prediction)}"

    # Standard KB search using FAISS
    D, I = kb_indexes[kb_name].search(q_vec.astype("float32"), top_k)
    return "\n".join(kb_facts[kb_name][i] for i in I[0])

# =========================
# Check if query is domain-specific
# =========================
"""
is_domain_question: function that checks if the query belongs to the hospital domain
query : the input question
threshold : probability cutoff for classification
return : boolean value indicating domain relevance
"""
def is_domain_question(query, threshold=0.5):
    vec = embedder.encode([query])[0]
    spark_row = spark.createDataFrame([Row(features=Vectors.dense(vec))])
    pred = model_re.transform(spark_row).collect()[0]
    return pred['probability'][1] > threshold

# =========================
# Chat loop
# =========================
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    # Determine context
    if is_domain_question(query, threshold=0.5):
        context = retrieve_context(query, top_k=1)
        # Prompt explicitly asks for one answer
        prompt = f"You are a helpful assistant.\nContext:\n{context}\nQuestion: {query}\nAnswer:"
    else:
        prompt = f"You are a helpful assistant.\nQuestion: {query}\nAnswer:"

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,               # limit response length
        do_sample=True,                   # enable sampling
        eos_token_id=tokenizer.eos_token_id,  # stop at EOS
        pad_token_id=tokenizer.eos_token_id
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove repeated input from the answer
    if answer.lower().startswith(prompt.lower()):
        answer = answer[len(prompt):].strip()

    # truncate at first double newline to avoid multi-Q/A generation
    answer = answer.split("\n\n")[0].strip()

    print("Answer:", answer, "\n")
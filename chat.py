from pyspark.sql import SparkSession, Row
from pyspark.ml.classification import LogisticRegressionModel
from pipeline_03_qlora import load_model


model_re =  LogisticRegressionModel.load(model_path)
model , kb_indexes, kb_facts = load_model()
spark = SparkSession.builder \
    .appName("DomainQuestionChecker") \
    .getOrCreate()
def get_kb_name(query):
    query = query.lower()
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
    # ===== Analytic KBs =====
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

    else:
        return None  # no KB for general question

def retrieve_context(query, top_k):
    """Retrieve top_k similar facts from a specific KB."""
    q_vec = embedder.encode([query], convert_to_numpy=True)
    kb_name = get_kb_name(query)
    if not kb_name:
        return "I don't know :(."
    ## knowledge base
    D, I = kb_indexes[kb_name].search(q_vec.astype("float32"), top_k)
    return "\n".join(kb_facts[kb_name][i] for i in I[0])
def is_domain_question(query, threshold):
    vec = embedder.encode([query])[0]
    from pyspark.ml.linalg import Vectors
    spark_row = spark.createDataFrame([Row(features=Vectors.dense(vec))])
    pred = model_re.transform(spark_row).collect()[0]
    return pred['probability'][1] > threshold

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    # ====== Decide whether to retrieve context ======
    if is_domain_question(query, threshold=0.5):
        context = retrieve_context(query ,top_k=1)
        prompt = f"""
Context:
{context}
Question: {query}

"""
    else:
        # Non-domain question: normal LLM response
        prompt = f"{query}\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.4,
        top_p=0.9
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Answer:", answer, "\n")

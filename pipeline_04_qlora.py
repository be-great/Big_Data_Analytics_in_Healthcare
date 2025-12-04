import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
from pyspark.sql import SparkSession
from datasets import Dataset

def load_model():
    base_file = ""
    # ====== CONFIG ======
    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_folder = "data/output/glora"
    ADAPTER_PATH = os.path.join(base_file, "data/output/adapters")  # Using os.path.join
    KB_PATHS = {
        "doctors": os.path.join(base_file, "data/output/kb/doctors"),
        "patients": os.path.join(base_file, "data/output/kb/patients"),
        "appointments": os.path.join(base_file, "data/output/kb/appointments"),
        "treatments": os.path.join(base_file, "data/output/kb/treatments"),
        "billing": os.path.join(base_file, "data/output/kb/billing"),
        "gender_comparison": os.path.join(base_file, "data/output/kb/gender_comparison"),
        "expensive_treatment": os.path.join(base_file, "data/output/kb/expensive_treatment"),
        "top_reasons_for_visits": os.path.join(base_file, "data/output/kb/top_reasons_for_visits"),
        "top_specializations": os.path.join(base_file, "data/output/kb/top_specializations"),
        "hospital_experience": os.path.join(base_file, "data/output/kb/hospital_experience"), # Added missing KB path
    }
    TOP_K = 3
    # ====== DEVICE ======
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)


    # ===== LOAD FAISS INDEXES AND FACTS =====
    kb_indexes = {}
    kb_facts = {}

    for kb_name, path in KB_PATHS.items():
        kb_indexes[kb_name] = faiss.read_index(f"{path}.index")
        kb_facts[kb_name] = np.load(f"{path}_facts.npy", allow_pickle=True)

    # ====== LOAD EMBEDDING MODEL ======
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


    # 1″ Load small dataset from Parquet
    spark = SparkSession.builder.appName("MakeSFTDataCPU").getOrCreate()
    #create_lightweight_training_examples
    def fmt(kb_paths_dict):
        """
        Generate lightweight LoRA training examples:
        one representative fact per KB to teach the model how to answer questions.
        """
        all_examples = []

        for kb_name, path in kb_paths_dict.items(): # Use the passed dictionary
            # Add a check to ensure the file exists before attempting to load
            facts_file = f"{path}_facts.npy"
            if not os.path.exists(facts_file):
                print(f"Warning: Knowledge base file not found for {kb_name}: {facts_file}. Skipping this KB.")
                continue

            facts = np.load(facts_file, allow_pickle=True)
            if len(facts) == 0:
                print(f"Warning: No facts found for {kb_name} in {facts_file}. Skipping this KB.")
                continue

            # Pick the first fact as representative
            fact = facts[0].strip()

            # Create instruction depending on KB type
            if kb_name == "patients":
                instr = "Tell me about the patient."
            elif kb_name == "doctors":
                instr = "Provide information about the doctor."
            elif kb_name == "appointments":
                instr = "Give details of the appointment."
            elif kb_name == "treatments":
                instr = "Explain the treatment."
            elif kb_name == "billing":
                instr = "Provide billing information."
            elif kb_name == "gender_comparison":
                instr = "Summarize gender visit distribution."
            elif kb_name == "expensive_treatment":
                instr = "Which treatment is the most expensive?"
            elif kb_name == "top_reasons_for_visits":
                instr = "What are the top reasons for visits?"
            elif kb_name == "top_specializations":
                instr = "What are the top medical specializations?"
            elif kb_name == "hospital_experience":
                instr = "Which hospital branches have the most experienced doctors?"
            else:
                instr = "Provide information."

            # Add the example in LoRA training format
            all_examples.append({
                "text": f"<s>[INSTRUCTION] {instr}\n[RESPONSE] {fact}</s>"
            })

        return all_examples

    # Corrected call to fmt function: pass KB_PATHS once
    train_data = fmt(KB_PATHS)
    dataset = Dataset.from_list(train_data)

    # 2″ Load base model normally (no quantization)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_name)
    base = prepare_model_for_kbit_training(base) 
    # 3″ LoRA config
    lconf = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"])
    model = get_peft_model(base, lconf)

    # 4″ Tokenization
    def tok(ex):
        out = tokenizer(ex["text"], truncation=True, padding="max_length", max_length=256)
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tok, batched=True, remove_columns=["text"])

    # 5″ Training (CPU-safe)
    train_args = TrainingArguments(
        output_dir=output_folder,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        use_cpu=True,      # ⬅️ ensures CPU only
        optim="adamw_torch" # Explicitly set optimizer to a valid option for CPU training
    )
    print("Starting training the model ")
    trainer = Trainer(model=model, args=train_args, train_dataset=tokenized)
    trainer.train()

    # 6″ Save adapter
    model = model.to('cpu') # Move model to CPU explicitly before saving
    model.save_pretrained(output_folder)
    tokenizer.save_pretrained(output_folder)
    print(" +++ LoRA adapter saved to:", output_folder)
    return model, kb_indexes, kb_facts, embedder, tokenizer

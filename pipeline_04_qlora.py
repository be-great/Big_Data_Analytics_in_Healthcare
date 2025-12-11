# -----------------------
# Purpose: load a base LLM model, integrate LoRA adapters, and build a GLORA model using hospital KBs.
#
# Objects:
#   - PyTorch & Transformers: for model loading and inference.
#   - PeftModel / LoRA adapters: lightweight fine-tuning for specific KB knowledge.
#   - SentenceTransformer: for embedding KB facts.
#   - FAISS: for storing embeddings for fast retrieval.
#   - SparkSession: used if needed to read Parquet datasets for lightweight examples.
#   - Dataset (HuggingFace): prepares LoRA training examples.
#   - KB_PATHS: dictionary of knowledge base paths (facts + FAISS indices).
#   - load_model(): loads base model, tokenizer, embedder, and FAISS indexes.
#   - create_glora_model(): prepares LoRA training data and trains LoRA adapters.
#
# Flow:
#   1) Define KB paths and output folders.
#   2) load_model():
#        - Load FAISS indexes and facts.
#        - Load embedding model (SentenceTransformer).
#        - Load base LLM + LoRA adapter.
#        - Return model, tokenizer, embedder, and KBs.
#   3) create_glora_model():
#        - Generate lightweight LoRA training examples from KBs.
#        - Tokenize examples.
#        - Configure LoRA and prepare base model for training.
#        - Train adapter on CPU (safe for low-resource setups).
#        - Save trained LoRA adapter and tokenizer.
# -----------------------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
from pyspark.sql import SparkSession
from datasets import Dataset

base_file = ""
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_folder = "data/output/glora"
ADAPTER_PATH = os.path.join(base_file, "data/output/glora")  # Using os.path.join
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
"""
load_model: function that loads base LLM, LoRA adapter, embeddings, and FAISS KBs
return: model, kb_indexes, kb_facts, embedder, tokenizer
"""
def load_model():
    # ====== CONFIG ======
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


    # ====== LOAD BASE + LoRA MODEL ON GPU ======
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map={"": "cpu"}  # Force CPU
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    
    return model, kb_indexes, kb_facts, embedder, tokenizer

"""
create_glora_model: function that creates LoRA adapter for GLORA
return: None (saves LoRA adapter and tokenizer)
"""
def create_glora_model():
    # 1″ Load small dataset from Parquet
    spark = SparkSession.builder.appName("MakeSFTDataCPU").getOrCreate()
    #create_lightweight_training_examples
    """
    fmt: function that formats training examples from KB facts
    kb_paths_dict: dictionary of KB names and paths
    return: list of dicts with 'text' for LoRA training
    """
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

    # 2- Load base model normally (no quantization)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_name)
    base = prepare_model_for_kbit_training(base) 
    # 3- LoRA config
    lconf = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"])
    model = get_peft_model(base, lconf)

    # 4- Tokenization
    """
    tok: function that tokenizes LoRA training examples
    ex: dictionary containing "text" key (LoRA training example)
    return: dictionary with tokenized input_ids, attention_mask, and labels
    """
    def tok(ex):
        out = tokenizer(ex["text"], truncation=True, padding="max_length", max_length=256)
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tok, batched=True, remove_columns=["text"])

    # 5- Training (CPU-safe)
    train_args = TrainingArguments(
        output_dir=output_folder,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        use_cpu=True,      # ensures CPU only
        optim="adamw_torch" # Explicitly set optimizer to a valid option for CPU training
    )
    print("Starting training the model ")
    trainer = Trainer(model=model, args=train_args, train_dataset=tokenized)
    trainer.train()

    # 6- Save adapter
    model = model.to('cpu') # Move model to CPU explicitly before saving
    model.save_pretrained(output_folder)
    tokenizer.save_pretrained(output_folder)
    print(" +++ LoRA adapter saved to:", output_folder)

"""uncomment if you want to create the glora model"""
# create_glora_model() 
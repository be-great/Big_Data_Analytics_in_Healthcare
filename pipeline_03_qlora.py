import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def load_model():
    base_file = ""
    # ====== CONFIG ======
    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ADAPTER_PATH = base_file + "data/output/adapters"  # adapter files should be inside
    KB_PATHS = {
        "doctors": base_file + "data/output/kb/doctors",
        "patients": base_file + "data/output/kb/patients",
        "appointments": base_file + "data/output/kb/appointments",
        "treatments": base_file + "data/output/kb/treatments",
        "billing": base_file + "data/output/kb/billing",
        "gender_comparison": base_file + "data/output/kb/gender_comparison",
        "expensive_treatment": base_file + "data/output/kb/expensive_treatment",
        "top_reasons_for_visits": base_file + "data/output/kb/top_reasons_for_visits",
        "treatments": base_file + "data/output/kb/treatments",
        "top_specializations": base_file + "data/output/kb/top_specializations",
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


    # ====== LOAD BASE + LoRA MODEL ON GPU ======
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optional: load in 4-bit for GPU memory saving
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",  # automatically puts model on GPU
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print("LoRA/Glora model loaded on GPU!")
    return model, kb_indexes, kb_facts

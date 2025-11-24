# Designing Localized Lightweight LLMsfor DomainsSpecific Application




## Literature research:-

### Optimizing Fine-Tuning in Quantized Language Models: An In-Depth Analysis of Key Variables
- methods 
- technologies
- results obtained
- suggestions 

- Topic: Accurate and efficient fine-tuning of quantized large language models via Balanced Low-Rank Adaptation (BLoRA).
- Methods:

    - Q‑BLoRA: balances the adapter input/output dimensions and adapter rank to address underfitting in quantized LLMs.

    - QA‑BLoRA: quantization-aware fine-tuning that aligns adapters with block-wise quantization, allowing merging back into the quantized model for inference efficiency.

- Technologies: LoRA, quantization (block-wise), applied on LLaMA, LLaMA2, Mistral, and Gemma models.
- suggestions/ Future Work :
    - Further exploration of the balancing factor λ for different models.
    - Extend to other quantization schemes and adapter merging strategies.
Link : https://www.techscience.com/cmc/v82n1/59229

## The Faiss library
 *The Faiss library is dedicated to vector similarity search, a core functionality of vector databases*

### FAISS for similarity search

#### Exact Search Methods (for exact nearest neighbor search)
- Flat (Brute‑Force)

#### Graph-Based ANN Methods (for fast approximate nearest neighbor search using graphs)
- HNSW
- NSG

#### Cluster-Based ANN Methods (for approximate nearest neighbor search using clustering)
- IVF (Inverted file) and its variants: IVF + PQ, IVF + SQ, IVF + OPQ, IVF + RQ/AQ, IVF + PQ + refinement ("+R")
#### Quantization-Based Methods (for memory-efficient vector compression)
- Scalar Quantization (SQ)
- Product Quantization (PQ)
- Optimized Product Quantization (OPQ)
- Residual / Additive Quantization (RQ / AQ)
- Multi-codebook Quantization

#### Hash-Based Methods (for fast search using hash-based vector grouping)
- LSH
- Binary Flat
- Binary IVF
- Binary HNSW
- Binary Hash / Multi-Hash

### Technologies
  - Hierarchical clustering and vector compression
  - Collection of C++ source files
  - Python wrapper
### results obtained
1. Faiss can handle million to billion-scale vector datasets
2. Faiss enables efficient similarity search over large embedding collections.
3. Balances speed, accuracy, and memory usage by using the above methods

### suggestions :
choose between different index types , and the optimal depends on teh problem's constraints.
that is why searcher should explor new index types.And depend on the problem domain 
we adjust parameters for trade-offs between speed, accuracy and memory.
Link : https://www.researchgate.net/publication/396647489_THE_FAISS_LIBRARY


### 3-title
- methods 
- technologies
- results obtained
- suggestions 
Link : https://link.springer.com/article/10.1007/s00778-025-00911-1


### 4-title
- methods 
- technologies
- results obtained
- suggestions 
Link : https://www.techscience.com/cmc/v82n1/59229

### 5-title
- methods 
- technologies
- results obtained
- suggestions 
Link : https://www.techscience.com/cmc/v82n1/59229



##  motivations



## technologies used



## data analysis methods 

## Findings obtained


## Concepts
1- parquet
2- faiss
3- glora
## Technologies :- 



## Enviroment: 

- NVIDIA "CUDA on WSL" 

## Files flow structure

1- setup_env.py : setup enviroment install pyspark and hadoop to the wsl
2- init.sh : create python env and install need python dependency and run the scripts
3- pipeline_01_data_process.py: process the big data by lower case , remove unwanted symobles and remove duplicates and save it as parquet
4- data_analysis.py : load parquets files Answer those scenarios:
    A — Which gender goes to the hospital more?
    B — Which hospital branch has the most experienced doctors?
    C — Which specialization dominates the others?
    D — What is the most common reason for visits?
    E — What is the ranking of treatments by cost?
5- pipeline_02_faiss.py: create the knowledge bases
6- pipeline_03_create_model.py: create the visitor prediction model
7- pipeline_04_qlora.py: adjust the model to domain specific
8- pipeline_05_domain_classifier.py: create a query classify model 
9- knowledge_evaluate.py: faiss knowledge base evlaution 
10- chat.py : the chat ai  
## Time processing 
- pyspark : 8 min

## future work :

1- add grammer correcter

## TODO
- Add the model output of the classify to estimated the next month coming visitor .

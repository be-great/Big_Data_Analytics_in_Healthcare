# Designing Localized Lightweight LLMsfor Domains Specific Application

### Abstract

As large language models (LLMs) continue to be adopted across a range of real-world applications, several ongoing challenges have become more noticeable, particularly those concerning scalability, data privacy, and computational cost. While high-end models such as GPT-4 and LLaMA-3 are known for their strong general performance, they often face difficulties when applied to focused domains or environments with limited computing resources. Our evaluation shows that LLMs can perform on par with considerably larger systems, yet offer advantages in inference speed, storage use, and the protection of sensitive information. Taken together, these results present a grounded pathway for developing compact and privacy-aware language models that can operate effectively.


### Introduction 

## Literature research

### 1. Optimizing Fine-Tuning in Quantized Language Models: An In-Depth Analysis of Key Variables
*Accurate and efficient fine-tuning of quantized large language models via Balanced Low-Rank Adaptation (BLoRA).*
### Methods:
- Q‑BLoRA: balances the adapter input/output dimensions and adapter rank to address underfitting in quantized LLMs.

- QA‑BLoRA: quantization-aware fine-tuning that aligns adapters with block-wise quantization, allowing merging back into the quantized model for inference efficiency.

### Technologies: 
- LoRA, quantization (block-wise), applied on LLaMA, LLaMA2, Mistral, and Gemma models.

### Results obtained:

- **Layer type matters:** MLP layers are robust to rank reduction; self-attention layers are more sensitive.  
- **Layer depth is less important:** Accuracy doesn’t vary significantly across different depths.  
- **Efficient adapters:** Memory usage can be greatly reduced without hurting performance.  
- **Parameter reduction strategy:** Reducing parameters in larger layers preserves accuracy better than in smaller layers.

### Suggestions For Future Work :
    
- Further exploration of the balancing factor for different models.
- Extend to other quantization schemes and adapter merging strategies.


- Article link: https://www.techscience.com/cmc/v82n1/59229

## 2. The Faiss library
 *The Faiss library is dedicated to vector similarity search, a core functionality of vector databases*

### Method for FAISS for similarity search

#### A. Exact Search Methods (for exact nearest neighbor search)
- Flat (Brute‑Force)

#### B. Graph-Based ANN Methods (for fast approximate nearest neighbor search using graphs)
- HNSW
- NSG

#### C. Cluster-Based ANN Methods (for approximate nearest neighbor search using clustering)
- IVF (Inverted file) and its variants: IVF + PQ, IVF + SQ, IVF + OPQ, IVF + RQ/AQ, IVF + PQ + refinement ("+R")
#### D. Quantization-Based Methods (for memory-efficient vector compression)
- Scalar Quantization (SQ)
- Product Quantization (PQ)
- Optimized Product Quantization (OPQ)
- Residual / Additive Quantization (RQ / AQ)
- Multi-codebook Quantization

#### E. Hash-Based
 Methods (for fast search using hash-based vector grouping)
- LSH
- Binary Flat
- Binary IVF
- Binary HNSW
- Binary Hash / Multi-Hash

### Technologies
  - Hierarchical clustering and vector compression
  - Collection of C++ source files
  - Python wrapper
### Results obtained
1. Faiss can handle million to billion-scale vector datasets
2. Faiss enables efficient similarity search over large embedding collections.
3. Balances speed, accuracy, and memory usage by using the above methods

### suggestions for future work:
Expanding the focus to include novel quantization techniques, better hardware support for some
indexe and new indexing forms, such as associative vector memories for transformer architectures 

- Article link: https://www.researchgate.net/publication/396647489_THE_FAISS_LIBRARY


### 3.Data formats in analytical DBMSs: performance trade-offs and future directions 
*Evaluates the suitability of three common formats (Apache Parquet, Apache ORC, and Apache Arrow) for use inside analytical DBMSs and Identifies features important for efficient querying (encoding, compression, data access) and explores trade‑offs between them.*
### Methods  
- Comparative analysis of the three formats across dimensions: compression ratio, transcoding throughput, data access cost, end‑to‑end query subexpressions.
- Benchmarks using real‑world datasets and analytic workloads (TPC‑DS, embedding/RAG datasets) to evaluate encoding, compression, and access performance.

### Technologies

- **Parquet (open columnar format):** uses dictionary encoding, run-length encoding, and bit-packing, and supports zone maps for efficient data skipping.  
- **ORC:** an on-disk columnar format optimized for read-heavy analytics, with bloom filters and fine-grained in-memory metadata for fast queries.  
- **Arrow:** an in-memory columnar format designed for fast inter-process communication (IPC) and analytics, but with less built-in encoding and compression.
### Results Obtained
- Parquet achieved superior compression: “reduce the size of the column data to about **13%** of the original” in one dataset.  
- ORC achieved ~27% of original size in the same test, Ar0row actually increased size by ~7% under default settings.  
- In data access (projection/filter), ORC often performed best for high selectivity queries, but Parquet performed better for very low selectivity due to fine‑grained skipping.
- Embedding‑dataset tests showed none of the formats fully optimal: “current formats do not adequately handle … embedding datasets.”
### Suggestions For Future Work

- Opportunity to **co‑design a unified in‑memory and on‑disk representation** rather than separate formats.  
- Improve support for **high‑dimensional, high‑entropy, high‑precision embeddings** in storage formats.  
- Further exploration of **workload‑aware format tuning (encoding/compression)** and pushing computation into the encoded domain.

- Article link: https://link.springer.com/article/10.1007/s00778-025-00911-1


### 4. A systematic review on big data applications and scope for industrial processing and healthcare sectors
*Reviews “data collection, analyzing, processing, and viewing” to explore big data in industrial processing and healthcare sectors.*

### Methods 
The authors conduct a systematic review based on the big-data life cycle, examining works related to “data collection,” “data integration,” “data preparation,” “data analysis,” and “data reuse.”
### Technologies
- machine learning, deep learning
- natural language generation
- metaheuristic algorithms such as “PSO” and the “firefly algorithm,”
- big-data frameworks like “Hadoop,” “Spark,” and “MapReduce.”
### Results obtained
The artical identifies major challenges such as "data cleaning and outlier analysis." It reports that big data can enhance
 - management (general management decision making)
 - patient-care services 
 - operational efficiency
### Suggestions for future work :
The authors propose developing “an optimization-based data cleaning model” and “an outlier removal model,” and recommend further research on scalable big-data platforms and advanced learning approaches including “unsupervised, semi-supervised, and streaming-based models.”
    
- Article link: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00808-2


### 5.  A Hybrid Retrieval-And-Generation Framework For Radiology Report Summarization With Faiss Indexing and T5 Transformers 
*A Retrieval-Augmented Generation (RAG) approach to medical report summarization by integrating a FAISS-based semantic search engine with a small T5 generative model.*

### Methods 
Use of Sentence-BERT to embed the “Findings” section of radiology reports from the MIMIC-III dataset, which are then indexed using FAISS to retrieve semantically similar cases.
### Technologies
- **FAISS** for approximate nearest-neighbor search  
- **T5 transformer** for text summarization  
- **Vector embeddings** of radiology report sections  
- Python-based machine learning libraries (**PyTorch**, **Hugging Face Transformers**)

### Results obtained
FAISS + T5 improves lexical fidelity, semantic consistency, and clinical relevance over traditional summarization methods, but some factual accuracy issues remain.
### Suggestions for future work :
- Integrating advanced retrieval techniques such as Dense Passage Retrieval (DPR), BM25 hybrid scoring, or domain-adaptive retrievers could yield more clinically coherent contextual inputs
- Leveraging larger encoder-decoder architectures or domain-pretrained models like BioBART or ClinicalT5 for further improve
summarization accuracy, especially in edge cases.
- Improve interpretability by integrating explanation modules, attention
heatmaps.
- Article link: https://dergipark.org.tr/en/pub/sdufenbed/issue/94267/1739565

  
  
  
  - *keywords*: parquet, faiss, glora

## Connection to Current project:-
This project aims to create a localized, lightweight LLM for domain-specific applications, with a case study on Healthcare Management. Parquet is used to compress data and speed up loading during processing. Faiss builds a vector-based knowledge base, enabling efficient retrieval without memorizing all data, reducing memory usage and improving scalability. QLoRA fine-tunes the LLM on domain-specific tasks using low-rank adapters, reducing both computation and memory requirements


##  Motivations



## Technologies used



## Methods 
![img](imgs/meth.png)
## Findings obtained


## Technologies :- 



## Enviroment: 

- NVIDIA "CUDA on WSL" 

## Files Flow Structure

1- **setup_env.py**: Set up environment, install PySpark and Hadoop on WSL.  
2- **init.sh**: Create Python environment, install needed Python dependencies, and run the scripts.  
3- **pipeline_01_data_process.py**: Process the big data by converting to lowercase, removing unwanted symbols, removing duplicates, and saving it as Parquet.  
4- **data_analysis.py**: Load Parquet files and answer the following scenarios:  
      A — Which gender goes to the hospital more?  
      B — Which hospital branch has the most experienced doctors?  
      C — Which specialization dominates the others?  
      D — What is the most common reason for visits?  
      E — What is the ranking of treatments by cost?  
5- **pipeline_02_faiss.py**: Create the knowledge bases.  
6- **pipeline_03_create_model.py**: Create the visitor prediction model.  
7- **pipeline_04_qlora.py**: Adjust the model to be domain-specific.  
8- **pipeline_05_domain_classifier.py**: Create a query classification model.  
9- **knowledge_evaluate.py**: FAISS knowledge base evaluation.  
10- **chat.py**: The chat AI.  
## Time processing 
- pyspark : 8 min

## Future work :

1- add grammer correcter

## TODO
- Add the model output of the classify to estimated the next month coming visitor .

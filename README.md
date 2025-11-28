# Designing Localized Lightweight LLMsfor Domains Specific Application

### Abstract

As large language models (LLMs) continue to be adopted across a range of real-world applications, several ongoing challenges have become more noticeable, particularly those concerning scalability, data privacy, and computational cost. While high-end models such as GPT-4 and LLaMA-3 are known for their strong general performance, they often face difficulties when applied to focused domains or environments with limited computing resources. Our evaluation shows that LLMs can perform on par with considerably larger systems, yet offer advantages in inference speed, storage use, and the protection of sensitive information. Taken together, these results present a grounded pathway for developing compact and privacy-aware language models that can operate effectively.


### Introduction 

## Literature research

### 1. Predictors of outpatients’ no-show: big data analytics using apache spark
healthcare organizations facing new opportunities, one of them is to improve the quality of healthcare. The main challenges is predictive analysis using techniques capable of handle the huge data generated.That is why a big data framework for identifying subject outpatients’ no-show been developed.
### Methods:
- Large-scale analysis of 2,011,813 outpatient appointments using distributed processing.

- Construction of predictive variables capturing behavioral, temporal, and operational characteristics.

- Application of Spark MLlib classifiers: Random Forest, Gradient Boosting, Logistic Regression, SVM, and MLP.

- Evaluation using train–test splits and 10-fold cross-validation.
### Technologies: 
- Apache Spark (parallel processing).

- Spark MLlib (classification pipeline).

- Hospital administrative datasets at million-record scale.

### Results obtained:

- Overall no-show rate: 26.7%.

- Primary predictors: prior no-shows, lead-time, prior attended visits, medical department.

- Secondary predictors: appointment type, clinic, month of appointment.

- Low-impact variables: gender, distance, nationality, reservation category.

- Gradient Boosting achieved the highest performance (≈ 79% accuracy, ≈ 0.81 AUC).

- Clear trade-off between computational efficiency and predictive performance across models.
### Suggestions For Future Work :
    
• Incorporating additional behavioral and contextual predictors
• Developing models for real-time no-show prediction within appointment systems
• Extending the analysis across diverse healthcare settings

- Article link:[ https://www.techscience.com/cmc/v82n1/59229](https://link.springer.com/article/10.1186/s40537-020-00384-9)

### 2. Applying Apache Spark on Streaming Big Data for Health Status Prediction
*Focuses on real-time health-status prediction using streaming IoT data combined with historical records.*

### Methods
- Real-time health-status prediction using Spark Streaming.
- Feature extraction from sensor streams and historical records.
- Machine-learning classification for health risk prediction.

### Technologies
- Apache Spark (streaming + batch processing).
- IoT sensor data.
- Spark-based machine learning modules.

### Results obtained
- Real-time processing of high-velocity health data.
- Timely alerts for potential health risks.
- Combining historical and streaming data improved prediction accuracy.

### Suggestions for future work
- Include more health indicators for broader prediction.
- Scale system to larger IoT deployments.
- Evaluate model performance across multiple institutions.

- Article link: [https://doi.org/10.32604/cmc.2022.019458](https://doi.org/10.32604/cmc.2022.019458)


### 3. Apache Spark for Analysis of Electronic Health Records: A Case Study of Diabetes Management
*Applies Spark to large-scale EHR data to improve diabetes prediction and management.*

### Methods
- Real EHR datasets processed with Apache Spark.  
- Data preprocessing, feature extraction, and ML model training.  

### Technologies
- Apache Spark (distributed + in-memory processing).  
- Machine-learning modules.  
- Electronic Health Records (EHR).  

### Results obtained
- Efficient processing of large EHR datasets.  
- Scalable analytics enabling diabetes prediction and management.  

### Suggestions for future work
- Data integration and privacy improvements.  
- Expanding datasets across multiple hospitals.  
- Enhancing model interpretability for clinical deployment.  

- Article link: https://doi.org/10.18280/ria.370616

### 4. Apache Spark in Healthcare: Advancing Data-Driven Innovations and Better Patient Care
*Explores Spark-based pipelines across multiple healthcare applications, including EHR analysis and predictive analytics.*

### Methods
- Case studies covering EHR management, predictive analytics, remote monitoring, and personalized medicine.  
- Spark-based pipelines for data preprocessing and ML/AI analysis.  

### Technologies
- Apache Spark (batch + streaming).  
- Healthcare datasets: EHR, imaging, remote monitoring.  
- ML/AI frameworks integrated with Spark.  

### Results obtained
- Efficient analysis of large-scale healthcare datasets.  
- Data-driven insights enabling better patient care.  
- Improved predictive analytics across multiple applications.  

### Suggestions for future work
- Real-time analytics and IoMT integration.  
- Data security, privacy, and interoperability.  
- Scalability and ethical compliance.  

- Article link: https://doi.org/10.14569/IJACSA.2023.0140665


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

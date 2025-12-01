# Designing Localized Lightweight LLMsfor Domains Specific Application
This project develops a lightweight, domain-specific AI system that integrates hospital visitor prediction, knowledge-base creation, and interactive querying to support efficient data-driven decision-making.
### Abstract

As large language models (LLMs) continue to be adopted across a range of real-world applications, several ongoing challenges have become more noticeable, particularly those concerning scalability, data privacy, and computational cost. While high-end models such as GPT-4 and LLaMA-3 are known for their strong general performance, they often face difficulties when applied to focused domains or environments with limited computing resources. Our evaluation shows that LLMs can perform on par with considerably larger systems, yet offer advantages in inference speed, storage use, and the protection of sensitive information. Taken together, these results present a grounded pathway for developing compact and privacy-aware language models that can operate effectively.


### Introduction 

## Literature research

### 1. Predictors of Outpatients’ No-show: Big Data Analytics using Apache Spark (2020)
*The study aims to identify key predictors of outpatient appointment no-shows using large-scale data analytics with Apache Spark.*

### Methods
- Large-scale analysis of 2,011,813 outpatient appointments.  
- Predictive variables capturing behavioral, temporal, and operational characteristics.  
- Applied Spark MLlib classifiers: Random Forest, Gradient Boosting, Logistic Regression, SVM, and MLP.  

### Technologies
- Apache Spark (parallel processing).  
- Spark MLlib.  
- Hospital administrative datasets at million-record scale.  

### Results obtained
- Overall no-show rate: 26.7%.  
- Top predictors: prior no-shows, lead-time, prior attended visits, medical department.  
- Secondary predictors: appointment type, clinic, month.  
- Gradient Boosting achieved highest performance (≈79% accuracy, ≈0.81 AUC).  

### Suggestions for future work
- Include additional behavioral and contextual predictors.  
- Develop real-time no-show prediction models.  
- Extend analysis across diverse healthcare settings.  

- Article link: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00384-9

### 2. Applying Apache Spark on Streaming Big Data for Health Status Prediction
*The work aims to build a real-time health-status prediction framework that processes streaming IoT data efficiently using Apache Spark.*

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
*The study aims to demonstrate how Apache Spark can enhance large-scale EHR analytics for improving diabetes prediction and management.*

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
*The goal is to review how Apache Spark enables scalable, data-driven analytics to improve healthcare operations and patient outcomes.*

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


### 5. Cloud-based Real-time SBP & HR Prediction using TCN and Apache Spark (2025)
*The research aims to build a real-time cloud-based framework for predicting systolic blood pressure and heart rate using Spark streaming and deep learning.*

### Methods
- Real-time SBP and HR prediction from streaming physiological data.  
- Multi-task Temporal Convolutional Network (TCN) compared with single-task models.  
- Data pipeline implemented with Apache Spark and Kafka.  

### Technologies
- Temporal Convolutional Network (TCN).  
- Apache Spark streaming.  
- MIMIC‑III dataset.  

### Results obtained
- Multi-task TCN outperforms single-task models (lower RMSE/MAE).  
- Scalable, low-latency, real-time monitoring achieved.  
- Demonstrates feasible integration of deep learning, streaming, and cloud/fog computing.  

### Suggestions for future work
- Extend to multi-parameter physiological monitoring.  
- Conduct real-world clinical validation.  
- Explore longer forecasting horizons and more complex models.  

- Article link: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-025-01207-5


## Connection to Current project:-
This project aims to create a localized, lightweight LLM for domain-specific applications, with a case study on Healthcare Management. Parquet is used to compress data and speed up loading during processing. Faiss builds a vector-based knowledge base, enabling efficient retrieval without memorizing all data, reducing memory usage and improving scalability. QLoRA fine-tunes the LLM on domain-specific tasks using low-rank adapters, reducing both computation and memory requirements.



##  Motivations
Before the rise of machine learning, earlier models struggled to understand the context of long sentences. To improve language understanding and create a general-purpose model, large language models (LLMs) were introduced. Businesses and sectors began to see new technological solutions to help them stay ahead of the curve and introduced new roles, such as the Prompt Engineer, to optimize LLM behavior. Therefore, when using external large language model (LLM) providers, companies faced new challenges regarding data privacy. That is why this project has been developed.




## Technologies :- 

- PySpark + Apache Spark: distributed data processing.
- Parquet: efficient storage format.
- FAISS: semantic knowledge retrieval.
- QLoRA: lightweight LLM fine-tuning
# Suggestions for future work
- Real-time visitor prediction with streaming data.
- Expand knowledge base across multiple hospital domains.
- Improve LLM interpretability and prediction accuracy.
- Utilize a catalog to enhance multimodal data organization by leveraging Neuralink’s data repository, which provides a unified and scalable way to structure, access, and manage diverse data types efficiently.


## Methods 
![img](imgs/meth.png)

## Questions that used for data analyzing:-
A — Which gender goes to the hospital more?
B — Which hospital branch has the most experienced doctors?
C — Which specialization dominates the others?
D — What is the most common reason for visits?
E — What is the ranking of treatments by cost?

Then with the help of RandomForestRegressor algorithm, we predict the expected number of visitors for the next month to help the healthcare management system prepare and allocate resources efficiently.
## Findings obtained
### Data analyzing:-
![gender](imgs/result_1_gender.png)
![experience](imgs/experience_doctor.png)
![special_dominates](imgs/speciliest_dominace.png)
![reason_for_visit](imgs/reason_for_visit.png)
![cost](imgs/treatment_cost.png)
### Number of visitors prediction 
![result2](imgs/result2.png)
## Enviroment: 

- NVIDIA "CUDA on WSL" 

## Files Flow Structure

1- **setup_env.py**: Set up environment, install PySpark and Hadoop on WSL.  
2- **init.sh**: Create Python environment, install needed Python dependencies, and run the scripts.  
3- **pipeline_01_data_process.py**: Process the big data by converting to lowercase, removing unwanted symbols, removing duplicates, and saving it as Parquet.  
4- **data_analysis.py**: Functions used for Loading Parquet files and answer the 5 scenarios questions
5- **pipeline_02_faiss.py**: Create the knowledge bases for each dataset and for the 5 scenarios
6- **knowledge_evaluate.py**: Evaluate the knowledge bases 
6- **pipeline_03_create_model.py**: Create the visitor prediction model.  
7- **pipeline_04_qlora.py**: Adjust the model to be domain-specific.  
8- **pipeline_05_domain_classifier.py**: Create a query classification model.  
9- **knowledge_evaluate.py**: FAISS knowledge base evaluation.  
10- **chat.py**: The chat AI.  
## Time processing 
- pyspark : 8 min

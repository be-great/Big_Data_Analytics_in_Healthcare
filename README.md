# Big_Data_Analytics_in_Healthcare









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

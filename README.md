# Project2_Disaster_Response_Pipeline

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Execution steps](#steps)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>
For this project I have used the disaster data provided by Udacity in order to develop a Machine learning Pipeline with the purpose of categorizing messages based on type and emmergency. 

![Screenshot 1](https://github.com/claudiaandreea/Project2_Disaster_Response_Pipeline/blob/main/DisasterResponseProject.png)

## File Descriptions <a name="files"></a>
ETL Pipeline Preparation.ipynb - create the ETL pipeline
ML Pipeline Preparation.ipynb -  create the ML pipeline

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md

## Execution steps <a name="steps"></a>
In order to run the model use the following commands:

For the process.py : python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
For the train_classifier : python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
For the run.py : python run.py
For the web app follow the link : http://0.0.0.0:3001/


## Licensing, Authors, Acknowledgements<a name="licensing"></a>
I would like to give credit to Udacity for the database used for this study and to for all the code techniques learning through the Data Science program. 
Feel free to use my code and maybe take a look at the blog post mentioned above. 
Github source: https://github.com/claudiaandreea/Project2_Disaster_Response_Pipeline 

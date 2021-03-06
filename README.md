# Project2_Disaster_Response_Pipeline

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Execution steps](#steps)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>
For this project I have used the disaster data provided by Udacity in order to categorize messages based on type and emmergency. The dataset containing real data based on pre-labelled tweets is used in developing a Machine learning Pipeline model that can make a big difference in health and security dillemas by generating alerts using different word triggers. The results are displayed in a web application that has been implemented as a real-time classification system that allows the user to imput a message and receive a predictive category. This projects is a starting point for my research in the field of health statistics, as I am a Phd student with most of the research in the telemedicine area. Conscientious about the project limitations I intend to extend this research using data from the telemedicine field. 

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
In order to run the model use the following commands.

1. Run the process.py script by: 
**python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db**

2. Run the train_classifier by: 
**python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl**

3. Run the run.py by: 
**python run.py**

4. Open the web app follow the link : http://0.0.0.0:3001/


## Licensing, Authors, Acknowledgements<a name="licensing"></a>
I would like to give credit to Udacity for the database used for this study and to for all the code techniques learning through the Data Science program. 
Feel free to use my code and maybe take a look at the blog post mentioned above. 
Github source: https://github.com/claudiaandreea/Project2_Disaster_Response_Pipeline 

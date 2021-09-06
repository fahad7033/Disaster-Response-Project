
# Disaster-Response-Project
<b>Overview of the project:  </b>

In this project, the disaster data from Figure Eight was analyzed to build and train a model for classifying messages.

ETL and machine learning pipelines were created as a procedure of data engineering, then the mode was trained on the processed data.


<b>Project Files: </b><br>
There are three main files in this project.

1-process_data.py ===> performs ETL Pipeline process, it Loads  messages dataset and categories dataset, and then Merges the two datasets and cleans the data then it stores it in a SQLite database

2-train_classifier.py ===> peforms ML Pipeline, it Loads data from the SQLite database Splits the dataset into training and test sets, then it builds a text processing and machine learning pipeline Trains and tunes a model using GridSearchCV Outputs results on the test set Exports the final model as a pickle file

3-run.py ===> Flask Web App which shows the data visualization and model's prediction.




<b> Installation: </b><br>
In order to install the required files, run the following commands to set your pipeline and model

1-To run ETL pipeline that cleans data and stores in database:
 python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

2-To run ML pipeline that trains classifier and save it:
 python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

3-To run your web app:
 python run.py



<b>Acknowledgment </b><br>
This project is a part of the Data Scientist Nanodegree Program offered by udacity (https://www.udacity.com/)

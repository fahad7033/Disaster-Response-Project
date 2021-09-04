
# Disaster-Response-Project
<b>Overview of the project  </b>

In this project, the disaster data from Figure Eight was analyzed to build and train a model for classifying messages.

ETL and machine learning pipelines were created as a procedure of data engineering, then the mode was trained on the processed data.

<b> To run the files: </b>
To run ETL pipeline that cleans data and stores in database:
 python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

To run ML pipeline that trains classifier and save it:
 python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

To run your web app:
 python run.py

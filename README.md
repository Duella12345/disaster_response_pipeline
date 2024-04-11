# Disaster Response Pipeline Project
![alt text](helicopter.jpg)
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores it in a sqlite database:
        
        `python process_data.py` (using defaults)
        
        `python process_data.py data/messages.csv data/categories.csv DisasterResponse.db` (full script)
    - To run ML pipeline that trains classifier and saves model:
        
        `python train_classifier.py` (using defaults)
        
        `python train_classifier.py DisasterResponse.db classifier.pkl`  (full script)

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

Screenshot of app:
![alt text](Screenshot_app.png)

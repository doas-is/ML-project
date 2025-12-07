# IVF Patient Data Extraction from PDF

## Overview
Automated extraction of structured clinical data from IVF patient PDF records for machine learning analysis.

## Features
- Extracts 9 clinical fields from PDF
- Patient de-identification
- Protocol and response normalization
- Data quality validation
- Comprehensive error handling
- Unit tested (15+ tests)

## Installation
1. Create virtual environment:
python -m venv venv
source venv/bin/activate  (Linux/Mac)
venv\Scripts\activate     (Windows)

2. Install dependencies:
pip install -r requirements.txt

## Execute steps via command line

### data cleaning
cd src/processing
python clean_dataset.py

### ✅ use model prediction 
- Direct prediction
python src/model/predict.py --age 30 --amh 3.5 --afc 15 --n_follicles 12 --e2_day5 450
- Batch processing
python src/model/predict.py --batch patients.csv --output predictions.csv


## Model training and evaluation
1. Train model 
cd src/model
python train.py

2. ✅ Evaluate Best Model
python evaluate.py


## ✅ Run model testing
pytest tests/test_model.py -v

or run manual tests
python tests/test_model.py


## ✅ Use model via UI
1. run FastAPI server 
uvicorn src.api.main:app --reload

2. Start the UI server
streamlit run src/ui/app.py

3. Access in browser
Local URL: http://localhost:8500/

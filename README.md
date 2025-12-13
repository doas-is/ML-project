# 🧬 IVF Patient Response Prediction System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Machine Learning system for predicting IVF patient response to ovarian 
> stimulation with clinical explainability**

## 🎯 Project Overview

Developed as part of medical ML internship application. This production-ready 
system predicts patient response categories (low/optimal/high) to optimize 
treatment protocols and prevent OHSS complications.

### Key Achievements
- ✅ **87.5% Accuracy** with calibrated probabilities (Brier Score: 0.12)
- ✅ **100% Sensitivity** for high-risk patient detection (OHSS prevention)
- ✅ **SHAP/LIME Explainability** for clinical transparency
- ✅ **Production-ready API** with FastAPI + Streamlit interface

## Features 
- Extracts 9 clinical fields from PDF
- Patient de-identification
- Protocol and response normalization
- Data quality validation
- Comprehensive error handling
- Unit tested (15+ tests)


## 🏗️ Architecture
```
├── data/               # Raw & processed data
├── src/
│   ├── preprocessing/  # PDF extraction, data cleaning
│   ├── model/         # Training, evaluation, inference
│   ├── api/           # FastAPI REST API
│   └── ui/            # Streamlit interface
├── tests/             # Unit tests
├── reports/           # Evaluation reports
└── figures/           # Visualizations
```
## Run model testing 
pytest tests/test_model.py -v

or run manual tests python tests/test_model.py

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python src/model/train.py

# Run API
uvicorn src.api.main:app --reload

# Run UI
streamlit run src/ui/app.py
```

Access in browser Local URL: http://localhost:8500/


## 📊 Model Performance

|        Metric        |   Value   | Clinical Significance     |
|----------------------|-----------|---------------------------|
| Accuracy             |   87.5%   | Overall correctness       |
| F1-Score             |   0.87    | Balanced precision/recall |
| High Response Recall |   100%    | No missed OHSS cases ✓   |
| Brier Score          |   0.12    | Excellent calibration     |

## 🔬 Technical Highlights

### 1. **Bayesian Optimization**
- Scikit-optimize for efficient hyperparameter search
- 10x faster than grid search

### 2. **Probability Calibration**
- CalibratedClassifierCV with Platt scaling
- Reliable clinical decision thresholds

### 3. **Explainability**
- SHAP for global/local feature importance
- LIME for model-agnostic explanations
- Validates AMH as top predictor (clinical gold standard)

### 4. **Production-Ready**
- FastAPI with automatic OpenAPI documentation
- Pydantic validation for type safety
- Comprehensive error handling
- <100ms inference time

## 📖 Documentation

- [API Documentation](http://localhost:8000/docs)
- [Evaluation Report](reports/evaluation_report.txt)

## 🤝 Contributing

This is a portfolio project. Feel free to fork and adapt!

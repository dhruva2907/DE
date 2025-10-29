# Sentiment Analysis Training Pipeline

Automated training pipeline for sentiment analysis models using Google Vertex AI.

## Overview
This pipeline automates:
1. Data loading and preprocessing
2. Feature engineering (TF-IDF)
3. Model training (scikit-learn)
4. Model evaluation
5. Model deployment to GCS

## Files
- `Pipeline.ipynb` - Notebook to create and run pipeline
- `Team14_FINAL_SUBMISSION.yaml` - Compiled pipeline definition

## Pipeline Components
1. **Data Loading** - Loads training data
2. **Preprocessing** - Cleans and prepares text
3. **Feature Engineering** - Creates TF-IDF features
4. **Training** - Trains sentiment classifier
5. **Evaluation** - Validates model performance
6. **Upload** - Saves model to GCS bucket

## Model Output
- **Bucket:** `model_mlops`
- **Model:** `Team-14-v2.pickle`

## Running Pipeline
See `Pipeline.ipynb` for execution instructions.

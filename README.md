# ML Repository

ML job scheduler delivering both scheduling via cron jobs and API endpoints for running ETL, training, prediction, evaluation, etc, of various end-to-end models. Developed with FastAPI, TensorFlow, PyTorch, AWS, MongoDB.

<br/>
<br/>

## Directory structure

The directory structure is as follows:

```
ml-job-scheduler/
├── api/
│   ├── controllers/
│   │   ├── btc_forecast_controller.py
│   │   └── __pycache__/
│   │       └── btc_forecast_controller.cpython-312.pyc
│   ├── models/
│   │   ├── btc_forecast/
│   │   │   ├── btc_forecast_models.py
│   │   │   └── __pycache__/
│   │   │       └── btc_forecast_models.cpython-312.pyc
│   │   ├── sp500_forecast/
│   │   ├── text_analyzer/
│   │   └── text_summarizer/
│   └── services/
│       ├── mongodb.py
│       └── __pycache__/
│           └── mongodb.cpython-312.pyc
├── btc_forecast.py
├── conf/
│   ├── aws/
│   ├── cron_jobs/
│   │   └── cron_jobs_conf.yml
│   └── mongodb/
│       ├── mongodb_conf.py
│       └── __pycache__/
│           └── mongodb_conf.cpython-312.pyc
├── data/
│   └── food_classifier/
│       └── models/
│           ├── keras_metadata.pb
│           └── saved_model.pb
├── Dockerfile
├── food_classifier.py
├── index.py
├── monitoring/
├── notebooks/
│   └── btc_forecast.py
├── README.md
├── requirements.txt
├── scripts/
│   ├── data_ops/
│   │   └── text_summarizer_data_ops.py
│   ├── etl/
│   │   ├── btc_forecast/
│   │   │   ├── btc_forecast_multivariate_etl.py
│   │   │   └── btc_forecast_univariate_etl.py
│   │   ├── food_classifier/
│   │   │   └── food_classifier_etl.py
│   │   ├── sp500_forecast/
│   │   │   ├── sp500_forecast_multivariate_etl.py
│   │   │   └── sp500_forecast_univariate_etl.py
│   │   ├── text_analyzer/
│   │   │   └── text_analyzer_etl.py
│   │   └── text_summarizer/
│   │       ├── text_summarizer_etl.py
│   │       ├── text_summarizer_positional_embedding.py
│   │       └── text_summarizer_tribid_embedding.py
│   ├── evaluation/
│   ├── full_pipeline/
│   │   ├── btc_forecast_multivariate_2_weeks.py
│   │   ├── btc_forecast_multivariate_current_day.py
│   │   ├── sp500_forecast_multivariate_2_weeks.py
│   │   └── __pycache__/
│   │       ├── btc_forecast_multivariate_2_weeks.cpython-312.pyc
│   │       └── btc_forecast_multivariate_current_day.cpython-312.pyc
│   ├── postprocessing/
│   ├── prediction/
│   │   └── text_analyzer_ensemble_prediction.py
│   ├── training/
│   │   └── text_analyzer_ensemble_training.py
│   └── training_prediction/
│       ├── btc_forecast_multivariate_training_prediction.py
│       ├── food_classifier_efficientb0_training_prediction.py
│       ├── sp500_forecast_multivariate_training_prediction.py
│       ├── text_summarizer_prediction_evaluation.py
│       └── text_summarizer_test_prediction.py
├── sp500_forecast.py
├── src/
│   └── mls/
│       ├── assets/
│       │   ├── btc_forecast_assets.py
│       │   ├── food_classifier_assets.py
│       │   ├── sp500_forecast_assets.py
│       │   └── __pycache__/
│       │       └── btc_forecast_assets.cpython-312.pyc
│       ├── data_ops/
│       │   ├── btc_forecast_load_prices.py
│       │   ├── sp500_load_prices.py
│       │   └── __pycache__/
│       │       └── btc_forecast_load_prices.cpython-312.pyc
│       ├── etl/
│       │   ├── btc_forecast_etl.py
│       │   ├── sp500_forecast_etl.py
│       │   ├── text_summarizer_etl.py
│       │   └── __pycache__/
│       │       └── btc_forecast_etl.cpython-312.pyc
│       ├── evaluation/
│       │   ├── btc_forecast_evaluation.py
│       │   ├── food_classifier_evaluation.py
│       │   ├── sp500_forecast_evaluation.py
│       │   ├── text_analyzer_evaluation.py
│       │   ├── text_summarizer_evaluation.py
│       │   └── __pycache__/
│       │       └── btc_forecast_evaluation.cpython-312.pyc
│       ├── model/
│       │   ├── btc_forecast_ensemble_model.py
│       │   ├── food_classifier_efficientb0_model.py
│       │   ├── sp500_forecast_ensemble_model.py
│       │   ├── text_analyzer_ensemble_model.py
│       │   ├── text_summarizer_tribid_embedding_model.py
│       │   └── __pycache__/
│       │       └── btc_forecast_ensemble_model.cpython-312.pyc
│       ├── postprocessing/
│       ├── prediction/
│       │   ├── btc_forecast_prediction.py
│       │   ├── sp500_forecast_prediction.py
│       │   └── __pycache__/
│       │       └── btc_forecast_prediction.cpython-312.pyc
│       └── training/
├── tests/
├── text_analyzer.py
├── text_summarizer.py
├── utils/
│   ├── api-requests/
│   ├── constants/
│   │   ├── btc_forecast_constants.py
│   │   └── __pycache__/
│   │       └── btc_forecast_constants.cpython-312.pyc
│   ├── errors/
│   └── helpers/
├── vercel.json
└── vercel_dev.json
```

<br/>
<br/>

## Overview

### Design

The usage of the service in other applications can be found below. Similar services can be found <a href="https://whimsical.com/web-microservices-6uqvwWZtcBFsNJB2hepGy1">here</a> and below:

#### Similar services

<figure>
  <img width="1000" alt="image" src="https://github.com/user-attachments/assets/80cf4ed7-58f7-4576-a3cd-d63ef584e0fd">
</figure>
Figure 1: High level view and usage in other applications

<br/>
<br/>

<img width="834" alt="image" src="https://github.com/user-attachments/assets/b54088e7-870c-46dd-9cf6-2e5ec27d9d5c">

<br/>
<br/>

### The ML job scheduler consists of the following models:

1. __BTC forecast__: Bitcoin forecasting
2. __S&P 500__: S&P 500 forecasting
3. __Food classifier__: Food detection / classification
4. __Text analyzer__: NLP text analysis
5. __Text summarizer__: NLP text summarization

### Running the jobs / models:

The jobs / models can be both manually triggered or scheduled via API calls or cron jobs respectively:

1. __BTC forecast__: <em>btc_forecast.py</em>
2. __S&P 500__: <em>sp500_forecast.py</em>
3. __Food classifier__: <em>food_classifier.py</em>
4. __Text analyzer__: <em>text_analyzer.py</em>
5. __Text summarizer__: <em>btc_summarizer.py</em>

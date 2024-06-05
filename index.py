from scripts.full_pipeline.btc_forecast_multivariate_current_day import main_btc_forecast_multivariate_current_day
from scripts.full_pipeline.btc_forecast_multivariate_2_weeks import main_btc_forecast_multivariate_2_weeks

from scripts.full_pipeline.sp500_forecast_multivariate_2_weeks import main_sp500_forecast_multivariate_2_weeks

from scripts.training.text_analyzer_ensemble_training import main_text_analyzer_training
from scripts.prediction.text_analyzer_ensemble_prediction import main_text_analyzer_prediction

from scripts.data_ops.text_summarizer_data_ops import *
from scripts.etl.text_summarizer import text_summarizer_etl, text_summarizer_positional_embedding, text_summarizer_tribid_embedding
from scripts.training_prediction import text_summarizer_prediction_evaluation, text_summarizer_test_prediction

from api.controllers.btc_forecast_controller import *

from fastapi import BackgroundTasks, FastAPI
from dotenv import load_dotenv
import os

load_dotenv()

# api routes for scheduled models
app = FastAPI()

# scheduling jobs
def schedule_btc_forecast_helper():
  # generate predictions
  future_forecast_current_day = main_btc_forecast_multivariate_current_day()
  future_furecast_2_weeks = main_btc_forecast_multivariate_2_weeks()

  # save predictions
  saveDailyPrediction(future_forecast_current_day)
  save2WeeksPrediction(future_furecast_2_weeks)

def schedule_sp500_forecast_helper():
  # generate predictions
  future_forecast_current_day = main_sp500_forecast_multivariate_2_weeks()

def schedule_text_analyzer_helper():
  # training and prediction
  main_text_analyzer_training()
  main_text_analyzer_prediction()

# routes
@app.post("/btc_forecast/{job_id}")
async def schedule_btc_forecast(job_id: str, background_tasks: BackgroundTasks):
  if job_id == os.getenv("BTC_FORECAST_DAILY_JOB_ID"):
    background_tasks.add_task(schedule_btc_forecast_helper)
    
    return {
      "message": "scheduled btc_forecast"
    }
  else:
    return {
      "message": "unable to schedule btc_forecast"
    }

@app.post("/sp500_forecast/{job_id}")
async def schedule_sp500_forecast(job_id: str, background_tasks: BackgroundTasks):
  if job_id == os.getenv("SP500_FORECAST_DAILY_JOB_ID"):
    background_tasks.add_task(schedule_sp500_forecast_helper)
    
    return {
      "message": "scheduled sp500_forecast"
    }
  else:
    return {
      "message": "unable to schedule sp500_forecast"
    }

@app.post("/text_analyzer/{job_id}")
async def schedule_text_analyzer(job_id: str, background_tasks: BackgroundTasks):
  if job_id == os.getenv("TEXT_ANALYZER_BI_WEEKLY_JOB_ID"):
    background_tasks.add_task(schedule_text_analyzer_helper)
    
    return {
      "message": "scheduled text_analyzer"
    }
  else:
    return {
      "message": "unable to schedule text_analyzer"
    }

@app.post("/text_summarizer/{job_id}")
async def schedule_text_summarizer(job_id: str, background_tasks: BackgroundTasks):
  if job_id == os.getenv("TEXT_SUMMARIZER_BI_WEEKLY_JOB_ID"):
    
    return {
      "message": "scheduled text_summarizer"
    }
  else:
    return {
      "message": "unable to schedule text_summarizer"
    }

@app.get("/test-route")
async def test_route():
  return {
    "message": "test route for ml-job-scheduler"
  }
from scripts.full_pipeline.btc_forecast_multivariate_current_day import main_btc_forecast_multivariate_current_day
from scripts.full_pipeline.btc_forecast_multivariate_2_weeks import main_btc_forecast_multivariate_2_weeks
from scripts.full_pipeline.sp500_forecast_multivariate_2_weeks import main_sp500_forecast_multivariate_2_weeks
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

@app.get("/test-route")
async def test_route():
  return {
    "message": "test route for ml-job-scheduler"
  }
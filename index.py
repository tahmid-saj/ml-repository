from scripts.full_pipeline.btc_forecast_multivariate_current_day import main_btc_forecast_multivariate_current_day
from scripts.full_pipeline.btc_forecast_multivariate_2_weeks import main_forecase_multivariate_2_weeks

from fastapi import BackgroundTasks, FastAPI
from dotenv import load_dotenv
import os

# api routes for scheduled models

app = FastAPI()

# scheduling jobs
def schedule_btc_forecast_helper():
  main_btc_forecast_multivariate_current_day()
  main_forecase_multivariate_2_weeks()

# routes
@app.get("/btc_forecast/{job_id}")
async def schedule_btc_forecast(job_id: str, background_tasks: BackgroundTasks):
  if job_id == os.getenv("BTC_FORECAST_DAILY_JOB_ID"):
    background_tasks.add_task(schedule_btc_forecast_helper)
    
    return {
      "message": "scheduled btc_forecast"
    }
  else:
    return {
      "message": "unable to schedule btc_forecase"
    }

@app.get("/test-route")
async def test_route():
  return {
    "message": "test route for ml-job-scheduler"
  }